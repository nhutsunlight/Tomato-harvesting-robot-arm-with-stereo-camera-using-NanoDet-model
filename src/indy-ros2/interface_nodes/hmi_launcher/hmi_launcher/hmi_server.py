#!/usr/bin/env python3
"""
hmi_server.py — Web server cầu nối giữa ROS2 và điện thoại/browser qua WiFi.

Chạy: python3 hmi_server.py
Truy cập: http://<IP_MÁY_TÍNH>:5000  (cùng mạng WiFi)

Yêu cầu: pip install flask flask-socketio pyyaml pillow
         (KHÔNG dùng eventlet — dùng threading mode)
"""

import os
import sys
import yaml
import time
import queue
import threading
import subprocess
import signal as _signal
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

# ── Optional ROS2 ──────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image as RosImage
    from start_msgs.msg import StartMsg
    import numpy as np
    import base64, io
    from PIL import Image as PILImage
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    np = None

from hmi_launcher.log_bus import bus

# ── Config paths (giống hmi_tkinter.py) ───────────────────────────────────────
def _find_config() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        candidate = parent / 'src' / 'indy-ros2' / 'msg' / 'config_manager' / 'config'
        if candidate.exists():
            return candidate
    return Path("/home/nhut/indy-ros2/config")

_CFG          = _find_config()
SETUP_PATH    = _CFG / 'setup.yaml'
DEPTH_PATH    = _CFG / 'stereo_config.yaml'
START_TOPIC   = "/start_msg"
IMG_DETECT    = "/stereo/left/image_yolo"
IMG_DEPTH     = "/stereo/depth/image_raw"
COORD_KW      = "Updated target:"
COLLECT_KW    = "collect_logger_node"
YOUR_NGROK_TOKEN = "2wXYIwNm9as3E0aFOpXlS0eAIOo_37MZEKmnDT1yivbh9Absi"

SETUP_LABELS = {
    "HomePose":"Home Pose (joint list)", "DorpPose":"Drop Pose (joint list)",
    "OffSetDistance":"Offset Distance (m)", "YOffSetDistance":"Y Offset Distance (m)",
    "OffSetAngle":"Offset Angle (rad)", "FxOffset":"Fx Offset (m)",
    "ObjectOffset":"Object Offset (m)", "Multi_collect_mode":"Multi Collect Mode (bool)",
}
STEREO_BM_LABELS = {
    "numDisparities":"Num Disparities","blockSize":"Block Size (odd≥1)",
    "preFilterType":"Pre-Filter Type","preFilterSize":"Pre-Filter Size",
    "preFilterCap":"Pre-Filter Cap","textureThreshold":"Texture Threshold",
    "uniquenessRatio":"Uniqueness Ratio (%)","speckleWindowSize":"Speckle Window",
    "speckleRange":"Speckle Range","disp12MaxDiff":"Disp12 Max Diff",
}
STEREO_SGBM_LABELS = {
    "minDisparity":"Min Disparity","numDisparities":"Num Disparities",
    "blockSize":"Block Size (odd≥1)","P1":"P1 (SGBM penalty)","P2":"P2 (SGBM penalty)",
    "disp12MaxDiff":"Disp12 Max Diff","uniquenessRatio":"Uniqueness Ratio (%)",
    "speckleWindowSize":"Speckle Window","speckleRange":"Speckle Range",
    "preFilterCap":"Pre-Filter Cap","mode":"Algorithm Mode",
}

# ── Flask / SocketIO — threading mode (không cần eventlet) ───────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'indy-hmi-secret'
sio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "ros_ok": False, "proc_ok": False,
    "log_coords": [], "log_collect": [],
    "img_detect": None, "img_depth": None,
}
_ros_node    = None
_ros_worker  = None
_start_pub   = None
_proc        = None
_proc_running = False
_camera_clients = set()

LOG_CHANNELS = ('log_coords', 'log_collect')

# ── Helpers ───────────────────────────────────────────────────────────────────
def _ts():
    return datetime.now().strftime("%H:%M:%S")

# ── Log queue — batch emit để tránh sio.emit() từ nhiều thread ──────────────
_log_queue = queue.Queue()

def _push_log(channel, line):
    """Thread-safe: đẩy vào queue, background thread sẽ emit."""
    if channel not in LOG_CHANNELS:
        return
    entry = f"[{_ts()}] {line}" if not line.startswith('[') else line
    _state[channel].append(entry)
    if len(_state[channel]) > 120:
        _state[channel] = _state[channel][-120:]
    _log_queue.put((channel, entry))

def _server_log(line):
    """Server-only log: print ra terminal, không đẩy lên web để tránh lag UI."""
    print(f"[{_ts()}] {line}", flush=True)

def _log_emitter():
    """Background thread: drain queue và emit theo batch."""
    while True:
        try:
            channel, entry = _log_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        # Gom thêm tối đa 20 dòng nữa trong cùng batch
        batch = [(channel, entry)]
        for _ in range(19):
            try:
                batch.append(_log_queue.get_nowait())
            except queue.Empty:
                break
        try:
            for ch, ln in batch:
                sio.emit('log', {'channel': ch, 'line': ln})
        except Exception:
            pass

def _load_yaml(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        return {"error": str(e)}

def _save_yaml(path, data):
    try:
        with open(path, 'w') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        return str(e)

# ── ROS2 ──────────────────────────────────────────────────────────────────────
class RosWorker:
    def __init__(self, node, executor):
        self.node = node
        self.executor = executor
        self.running = True
        self.last_img = {}
        self.min_interval = 0.2

    def spin(self):
        while self.running:
            try: self.executor.spin_once(timeout_sec=0.05)
            except: break

    def stop(self):
        self.running = False
        try: self.executor.shutdown()
        except: pass

    def encode_img(self, topic, msg):
        if not _camera_clients:
            return
        now = time.time()
        if now - self.last_img.get(topic, 0) < self.min_interval:
            return
        self.last_img[topic] = now
        try:
            enc = msg.encoding.lower()
            raw = np.frombuffer(msg.data, dtype=np.uint8).copy()
            if enc == "mono8":
                arr = raw.reshape((msg.height, msg.width))
                arr = np.stack([arr]*3, axis=2)
            elif enc == "bgr8":
                arr = raw.reshape((msg.height, msg.width, 3))[:,:,::-1].copy()
            else:
                arr = raw.reshape((msg.height, msg.width, 3))
            # resize for mobile bandwidth
            img = PILImage.fromarray(arr).resize((320, 240), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode()
            key = 'img_detect' if topic == IMG_DETECT else 'img_depth'
            _state[key] = b64
            for sid in list(_camera_clients):
                sio.emit('image', {'channel': key, 'data': b64}, to=sid)
        except Exception as e:
            pass

def _init_ros():
    global _ros_node, _ros_worker, _start_pub
    if not HAS_ROS:
        _server_log('[SERVER] rclpy not found - ROS disabled')
        return
    try:
        rclpy.init(args=None)
        _ros_node  = rclpy.create_node("hmi_web_node")
        _start_pub = _ros_node.create_publisher(StartMsg, START_TOPIC, 10)
        executor   = SingleThreadedExecutor()
        executor.add_node(_ros_node)
        _ros_worker = RosWorker(_ros_node, executor)
        _ros_node.create_subscription(RosImage, IMG_DETECT,
            lambda m: _ros_worker.encode_img(IMG_DETECT, m), 1)
        _ros_node.create_subscription(RosImage, IMG_DEPTH,
            lambda m: _ros_worker.encode_img(IMG_DEPTH, m), 1)
        t = threading.Thread(target=_ros_worker.spin, daemon=True)
        t.start()
        _state['ros_ok'] = True
        _server_log('[SERVER] ROS2 node ready')
        try:
            sio.emit('status', {'ros': True})
        except Exception:
            pass
    except Exception as e:
        _server_log(f'[SERVER] ROS2 init error: {e}')

# ── Process (AUTO_CMD) ────────────────────────────────────────────────────────
def _read_proc(proc):
    global _proc_running
    try:
        while _proc_running and proc:
            line = proc.stdout.readline()
            if not line: break
            line = line.strip()
            if len(line) > 400: line = line[:400] + '...'
            low = line.lower()
            if COORD_KW.lower() in low:
                _push_log('log_coords', line)
            elif COLLECT_KW.lower() in low:
                _push_log('log_collect', line)
    except: pass
    finally:
        _server_log('[SERVER] Process ended')
        _state['proc_ok'] = False
        sio.emit('status', {'proc': False})

# ── HTTP Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_APP)

@app.route('/api/config/<name>')
def api_config_get(name):
    if name == 'setup':
        raw = _load_yaml(SETUP_PATH)
        data = raw.get('setup', raw)
        return jsonify({'data': data, 'labels': SETUP_LABELS})
    elif name == 'depth':
        raw  = _load_yaml(DEPTH_PATH)
        meth = raw.get('stereo_method', 'bm')
        data = raw.get(f'stereo_{meth}', {})
        lbls = STEREO_SGBM_LABELS if meth == 'sgbm' else STEREO_BM_LABELS
        return jsonify({'data': data, 'labels': lbls, 'method': meth})
    return jsonify({'error': 'unknown'}), 404

@app.route('/api/config/<name>', methods=['POST'])
def api_config_set(name):
    body = request.json or {}
    if name == 'setup':
        raw = _load_yaml(SETUP_PATH)
        sec = raw.setdefault('setup', {})
        sec.update(body)
        err = _save_yaml(SETUP_PATH, raw)
        _server_log(f'[SETUP] Saved {list(body.keys())}')
    elif name == 'depth':
        raw  = _load_yaml(DEPTH_PATH)
        meth = body.pop('stereo_method', raw.get('stereo_method', 'bm'))
        raw['stereo_method'] = meth
        raw.setdefault(f'stereo_{meth}', {}).update(body)
        err = _save_yaml(DEPTH_PATH, raw)
        _server_log(f'[DEPTH] Saved method={meth} {list(body.keys())}')
    else:
        return jsonify({'ok': False, 'error': 'unknown'}), 404
    return jsonify({'ok': err is True, 'error': err if err is not True else None})

@app.route('/api/logs')
def api_logs():
    return jsonify({k: _state[k] for k in LOG_CHANNELS})

@app.route('/api/images')
def api_images():
    return jsonify({k: _state[k] for k in ('img_detect','img_depth')})

# ── SocketIO events ───────────────────────────────────────────────────────────
@sio.on('connect')
def on_connect():
    emit('status', {
        'ros':  _state['ros_ok'],
        'proc': _state['proc_ok'],
    })
    # send cached logs — merge _state (server logs) với bus snapshot (HMI logs)
    try:
        bus_snap = bus.snapshot()
    except Exception:
        bus_snap = {}
    for ch in LOG_CHANNELS:
        lines = _state.get(ch, [])[-25:] + bus_snap.get(ch, [])[-25:]
        seen = set()
        for line in lines:
            if line not in seen:
                seen.add(line)
                emit('log', {'channel': ch, 'line': line})

@sio.on('command')
def on_command(data):
    global _proc, _proc_running
    cmd = data.get('cmd')

    if cmd == 'start':
        if HAS_ROS and _start_pub:
            try:
                msg = StartMsg(); msg.start = True
                _start_pub.publish(msg)
                _server_log('[HMI] Published start=True')
                emit('status', {'started': True})
            except Exception as e:
                _server_log(f'[HMI] Publish error: {e}')
        else:
            _server_log('[HMI] ROS unavailable')

    elif cmd == 'stop':
        if HAS_ROS and _start_pub:
            try:
                msg = StartMsg(); msg.start = False
                _start_pub.publish(msg)
                _server_log('[HMI] Published start=False')
                emit('status', {'started': False})
            except Exception as e:
                _server_log(f'[HMI] Publish error: {e}')

    elif cmd == 'launch':
        auto_cmd = data.get('command', '').strip()
        if not auto_cmd:
            emit('error', {'msg': 'No command provided'}); return
        if _proc and _proc.poll() is None:
            emit('error', {'msg': 'Process already running'}); return
        try:
            _proc_running = True
            _proc = subprocess.Popen(
                auto_cmd, shell=True, executable='/bin/bash',
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, preexec_fn=os.setsid)
            threading.Thread(target=_read_proc, args=(_proc,), daemon=True).start()
            _state['proc_ok'] = True
            _server_log(f'[SERVER] Launched PID {_proc.pid}')
            sio.emit('status', {'proc': True})
        except Exception as e:
            _server_log(f'[SERVER] Launch error: {e}')

    elif cmd == 'kill':
        _proc_running = False
        if _proc:
            try:
                os.killpg(os.getpgid(_proc.pid), _signal.SIGTERM)
                _server_log('[SERVER] Process killed')
            except: pass
        _state['proc_ok'] = False
        sio.emit('status', {'proc': False})

    elif cmd == 'clear_logs':
        for ch in LOG_CHANNELS:
            _state[ch].clear()
        sio.emit('clear_logs', {})

@sio.on('view')
def on_view(data):
    sid = request.sid
    if data.get('tab') == 'cam':
        _camera_clients.add(sid)
        for k in ('img_detect','img_depth'):
            if _state[k]:
                emit('image', {'channel': k, 'data': _state[k]})
    else:
        _camera_clients.discard(sid)

@sio.on('disconnect')
def on_disconnect():
    _camera_clients.discard(request.sid)

# ── HTML App (single-file PWA) ────────────────────────────────────────────────
HTML_APP = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0d0f14">
<title>INDY ROS2 HMI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
:root {
  --bg:      #0d0f14;
  --panel:   #131720;
  --card:    #161b26;
  --card-s:  #1a2235;
  --border:  #1e2840;
  --accent:  #00e5ff;
  --green:   #39ff7e;
  --red:     #ff3b5c;
  --yellow:  #ffd166;
  --text:    #c8d6e5;
  --dim:     #4a5568;
  --sub:     #7a8fa8;
  --kbd:     #0e1118;
  --key:     #1c2333;
  --editor:  #10131a;
  --font:    'Courier New', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); overflow: hidden; }

/* ── Top nav bar ── */
#topbar {
  display: flex; align-items: center; justify-content: space-between;
  background: #0a0c11; height: 52px; padding: 0 16px;
  border-bottom: 2px solid var(--accent); flex-shrink: 0;
}
#topbar .brand { display: flex; align-items: center; gap: 8px; }
#topbar .brand span.hex { color: var(--accent); font-size: 20px; }
#topbar .brand span.title { color: var(--accent); font-weight: bold; font-size: 15px; }
#topbar .brand span.status { color: var(--dim); font-size: 11px; margin-left: 8px; }
#topbar .dots { display: flex; gap: 8px; align-items: center; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); transition: background .3s; }
.dot.on { background: var(--green); }
.dot.err { background: var(--red); }

/* ── Tab bar ── */
#tabs {
  display: flex; background: #0a0c11;
  border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.tab {
  flex: 1; padding: 10px 4px; text-align: center;
  font-size: 11px; font-weight: bold; color: var(--dim);
  cursor: pointer; border-bottom: 2px solid transparent;
  transition: all .2s; text-transform: uppercase; letter-spacing: .05em;
}
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ── Pages ── */
#app { display: flex; flex-direction: column; height: 100vh; }
.page { display: none; flex: 1; overflow: hidden; flex-direction: column; }
.page.active { display: flex; }

/* ── Control page ── */
#page-ctrl { padding: 0; }
.ctrl-scroll { overflow-y: auto; flex: 1; padding: 16px; }
.section-label { color: var(--dim); font-size: 10px; font-weight: bold; letter-spacing: .1em; margin-bottom: 8px; margin-top: 16px; }
.section-label:first-child { margin-top: 0; }
.btn-row { display: flex; gap: 10px; margin-bottom: 4px; }
.btn {
  flex: 1; padding: 16px 12px; font-family: var(--font); font-size: 14px; font-weight: bold;
  border: none; border-radius: 4px; cursor: pointer; transition: all .15s; letter-spacing: .03em;
}
.btn-green  { background: #1a3a1f; color: var(--green); }
.btn-red    { background: #3a1a1a; color: var(--red); }
.btn-yellow { background: #2a2010; color: var(--yellow); }
.btn-blue   { background: #111c26; color: var(--accent); }
.btn-dim    { background: var(--border); color: var(--text); }
.btn:active { opacity: .7; transform: scale(.97); }
.status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.status-item {
  background: var(--card); border-radius: 4px; padding: 12px;
  display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--sub);
}
.status-item .dot { flex-shrink: 0; }

/* ── QR overlay ── */
#qr-overlay {
  display: none; position: fixed; inset: 0; z-index: 9999;
  background: rgba(0,0,0,.82);
  align-items: center; justify-content: center; flex-direction: column; gap: 16px;
}
#qr-overlay.show { display: flex; }
#qr-overlay img  { width: 220px; height: 220px; border-radius: 8px; }
#qr-overlay .qr-url {
  color: var(--accent); font-size: 13px; font-weight: bold;
  font-family: var(--font); text-align: center; padding: 0 20px;
}
#qr-overlay .qr-close {
  background: var(--border); color: var(--text); font-family: var(--font);
  font-size: 13px; font-weight: bold; border: none; border-radius: 6px;
  padding: 10px 28px; cursor: pointer; margin-top: 4px;
}
#qr-overlay .qr-close:active { opacity: .7; }

/* ── Cameras ── */
#page-cam { gap: 4px; padding: 4px; }
.cam-box { flex: 1; background: #050608; border-radius: 4px; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; }
.cam-box img { width: 100%; height: 100%; object-fit: contain; display: block; }
.cam-label {
  position: absolute; top: 6px; left: 8px;
  font-size: 10px; font-weight: bold; color: var(--accent);
  background: rgba(0,0,0,.6); padding: 2px 6px; border-radius: 3px;
}
.cam-fps {
  position: absolute; top: 6px; right: 8px;
  font-size: 10px; color: var(--dim);
  background: rgba(0,0,0,.6); padding: 2px 6px; border-radius: 3px;
}
.cam-placeholder { color: var(--dim); font-size: 12px; }

/* ── Logs ── */
#page-log { gap: 0; }
.log-tabs { display: flex; background: #0a0c11; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.log-tab { flex: 1; padding: 8px 4px; text-align: center; font-size: 10px; font-weight: bold; color: var(--dim); cursor: pointer; border-bottom: 2px solid transparent; }
.log-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.log-content { flex: 1; overflow-y: auto; padding: 8px; font-size: 11px; line-height: 1.6; }
.log-line { padding: 2px 0; border-bottom: 1px solid rgba(30,40,64,.4); word-break: break-all; }
.log-line.coord { color: var(--green); }
.log-line.collect { color: var(--yellow); }

/* ── Setup list screen ── */
.param-list { overflow-y: auto; flex: 1; }
.param-row {
  display: flex; align-items: stretch; background: var(--card);
  border-bottom: 1px solid var(--border); cursor: pointer;
  transition: background .15s; min-height: 64px;
}
.param-row:active, .param-row.sel { background: var(--card-s); }
.param-row .accent-bar { width: 4px; background: var(--card); flex-shrink: 0; transition: background .15s; }
.param-row.sel .accent-bar { background: var(--accent); }
.param-row .inner { flex: 1; padding: 12px 14px; min-width: 0; }
.param-row .top-line { display: flex; align-items: center; gap: 6px; }
.param-row .idx { color: var(--dim); font-size: 11px; flex-shrink: 0; }
.param-row .lbl { flex: 1; font-size: 14px; font-weight: bold; color: var(--text); }
.badge { font-size: 10px; font-weight: bold; color: #0d0f14; padding: 2px 6px; border-radius: 3px; flex-shrink: 0; }
.param-row .bot-line { display: flex; justify-content: space-between; align-items: center; margin-top: 4px; }
.param-row .key-name { font-size: 11px; color: var(--dim); }
.param-row .val { font-size: 13px; font-weight: bold; }

/* ── Setup editor screen ── */
#editor { display: none; flex-direction: column; flex: 1; overflow: hidden; }
#editor.active { display: flex; }
.editor-form { flex: 1; overflow-y: auto; padding: 16px 20px; }
.back-btn {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--border); color: var(--text); font-family: var(--font);
  font-size: 13px; font-weight: bold; border: none; border-radius: 4px;
  padding: 10px 16px; cursor: pointer; margin-bottom: 14px;
}
.back-btn:active { opacity: .7; }
.editor-title { color: var(--accent); font-size: 16px; font-weight: bold; }
.editor-key   { color: var(--dim); font-size: 12px; margin-left: 6px; }
.curr-pill {
  background: var(--card); border-radius: 4px; padding: 12px 14px;
  margin: 10px 0 14px; border-left: 4px solid var(--accent);
}
.curr-pill .cl { color: var(--dim); font-size: 10px; font-weight: bold; }
.curr-pill .cv { color: var(--yellow); font-size: 14px; font-weight: bold; margin-top: 4px; word-break: break-all; }
.input-lbl { color: var(--sub); font-size: 12px; margin-bottom: 8px; }
.input-wrap { background: var(--accent); padding: 2px; border-radius: 4px; margin-bottom: 14px; }
.input-wrap input {
  width: 100%; background: var(--card); color: var(--accent);
  font-family: var(--font); font-size: 17px; font-weight: bold;
  border: none; padding: 12px; outline: none; border-radius: 3px;
}
.save-btn {
  background: var(--green); color: #0b0d12; font-family: var(--font);
  font-size: 14px; font-weight: bold; border: none; border-radius: 4px;
  padding: 14px 26px; cursor: pointer; width: 100%; margin-top: 4px;
}
.save-btn:active { opacity: .8; }
.bool-row { display: flex; gap: 12px; margin-bottom: 14px; }
.bool-btn {
  flex: 1; padding: 16px; font-family: var(--font); font-size: 15px; font-weight: bold;
  border: 2px solid transparent; border-radius: 6px; cursor: pointer; background: var(--card);
  transition: all .2s;
}
.bool-btn.sel-on  { border-color: var(--green); color: var(--green); }
.bool-btn.sel-off { border-color: var(--red); color: var(--red); }
.bool-btn.on  { color: var(--green); }
.bool-btn.off { color: var(--red); }

/* ── Virtual keyboard ── */
#vkbd {
  background: var(--kbd); padding: 8px 10px 12px; flex-shrink: 0;
  border-top: 1px solid var(--border);
}
.kbd-row { display: flex; justify-content: center; gap: 4px; margin-bottom: 4px; }
.kkey {
  background: var(--key); color: var(--text); font-family: var(--font);
  font-size: 13px; font-weight: bold; border: none; border-radius: 4px;
  min-width: 26px; height: 36px; padding: 0 4px; cursor: pointer; flex: 1; max-width: 34px;
  transition: background .1s;
}
.kkey:active { background: var(--accent); color: #0b0d12; }
.kkey.sp { max-width: none; flex: none; font-size: 11px; color: var(--sub); padding: 0 8px; }
.kkey.space-key { flex: 4; max-width: 140px; }

/* ── Depth method bar ── */
.method-bar { display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #0a0d14; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.method-bar .ml { color: var(--dim); font-size: 11px; font-weight: bold; }
.method-btn { background: var(--key); color: var(--sub); font-family: var(--font); font-size: 12px; font-weight: bold; border: none; border-radius: 4px; padding: 8px 18px; cursor: pointer; transition: all .2s; }
.method-btn.active { background: var(--accent); color: #0d0f14; }

/* ── Toast ── */
#toast {
  position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%);
  background: var(--card-s); color: var(--text); font-size: 13px; font-weight: bold;
  padding: 10px 20px; border-radius: 20px; border: 1px solid var(--border);
  display: none; z-index: 999; white-space: nowrap;
}
</style>
</head>
<body>

<!-- QR overlay -->
<div id="qr-overlay" onclick="closeQR()">
  <img id="qr-img" src="" alt="QR Code">
  <div class="qr-url" id="qr-url"></div>
  <button class="qr-close" onclick="closeQR()">✕  CLOSE</button>
</div>

<div id="app">

  <!-- Top bar -->
  <div id="topbar">
    <div class="brand">
      <span class="hex">⬡</span>
      <span class="title">INDY ROS2</span>
      <span class="status" id="sys-status">CONNECTING…</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px">
      <button onclick="showQR()" style="background:none;border:1px solid var(--border);
        color:var(--accent);font-family:var(--font);font-size:11px;font-weight:bold;
        padding:5px 10px;border-radius:4px;cursor:pointer">QR</button>
      <div class="dots">
        <div class="dot" id="dot-ros"  title="ROS2"></div>
        <div class="dot" id="dot-proc" title="Process"></div>
        <div class="dot" id="dot-net"  title="Network"></div>
      </div>
    </div>
  </div>

  <!-- Tab bar -->
  <div id="tabs">
    <div class="tab active" onclick="switchTab('ctrl')">⚡ Control</div>
    <div class="tab" onclick="switchTab('cam')">📷 Camera</div>
    <div class="tab" onclick="switchTab('log')">📋 Log</div>
    <div class="tab" onclick="switchTab('setup')">⚙ Setup</div>
    <div class="tab" onclick="switchTab('depth')">🔭 Depth</div>
  </div>

  <!-- ── CONTROL ── -->
  <div class="page active" id="page-ctrl">
    <div class="ctrl-scroll">
      <div class="section-label">ROBOT CONTROL</div>
      <div class="btn-row">
        <button class="btn btn-green" onclick="sendCmd('start')">▶  START</button>
        <button class="btn btn-red"   onclick="sendCmd('stop')">■  STOP</button>
      </div>

      <div class="section-label">CONFIGURATION</div>
      <button class="btn btn-yellow" style="width:100%;margin-bottom:8px" onclick="switchTab('setup')">⚙  SETUP PARAMETERS</button>
      <button class="btn btn-yellow" style="width:100%;margin-bottom:8px" onclick="switchTab('depth')">⚙  DEPTH SETUP</button>

      <div class="section-label">LOGS</div>
      <button class="btn btn-blue" style="width:100%;margin-bottom:8px" onclick="switchTab('log')">📋  VIEW LOGS</button>
      <button class="btn btn-dim"  style="width:100%;margin-bottom:8px" onclick="sendCmd('clear_logs')">🗑  CLEAR ALL LOGS</button>

      <div class="section-label">SYSTEM STATUS</div>
      <div class="status-grid">
        <div class="status-item"><div class="dot" id="s-ros"></div> ROS2 Node</div>
        <div class="status-item"><div class="dot" id="s-proc"></div> Launch Proc</div>
        <div class="status-item"><div class="dot" id="s-net" style="background:var(--green)"></div> WiFi</div>
        <div class="status-item"><div class="dot" id="s-pub"></div> Publisher</div>
      </div>
    </div>
  </div>

  <!-- ── CAMERAS ── -->
  <div class="page" id="page-cam">
    <div class="cam-box">
      <img id="img-detect" style="display:none">
      <div class="cam-placeholder" id="ph-detect">● no signal</div>
      <div class="cam-label">DETECT</div>
      <div class="cam-fps" id="fps-detect">–</div>
    </div>
    <div class="cam-box">
      <img id="img-depth" style="display:none">
      <div class="cam-placeholder" id="ph-depth">● no signal</div>
      <div class="cam-label">DEPTH</div>
      <div class="cam-fps" id="fps-depth">–</div>
    </div>
  </div>

  <!-- ── LOGS ── -->
  <div class="page" id="page-log">
    <div class="log-tabs">
      <div class="log-tab active" onclick="switchLogTab('log_main')">MAIN</div>
      <div class="log-tab" onclick="switchLogTab('log_coords')">COORDS</div>
      <div class="log-tab" onclick="switchLogTab('log_collect')">COLLECT</div>
    </div>
    <div class="log-content" id="log-view"></div>
  </div>

  <!-- ── SETUP ── -->
  <div class="page" id="page-setup">
    <!-- list -->
    <div id="setup-list" style="display:flex;flex-direction:column;flex:1;overflow:hidden">
      <div class="param-list" id="setup-rows"></div>
    </div>
    <!-- editor -->
    <div id="editor" id="setup-editor">
      <div class="editor-form" id="editor-form"></div>
      <div id="vkbd" style="display:none"></div>
    </div>
  </div>

  <!-- ── DEPTH ── -->
  <div class="page" id="page-depth">
    <div class="method-bar">
      <span class="ml">ALGORITHM:</span>
      <button class="method-btn active" id="btn-bm"   onclick="setMethod('bm')">BM</button>
      <button class="method-btn"        id="btn-sgbm" onclick="setMethod('sgbm')">SGBM</button>
    </div>
    <div id="depth-list" style="display:flex;flex-direction:column;flex:1;overflow:hidden">
      <div class="param-list" id="depth-rows"></div>
    </div>
    <div id="depth-editor" style="display:none;flex-direction:column;flex:1;overflow:hidden">
      <div class="editor-form" id="depth-editor-form"></div>
      <div id="depth-vkbd" style="display:none"></div>
    </div>
  </div>

</div>
<div id="toast"></div>

<script>
// ── Socket ──────────────────────────────────────────────────────────────────
const socket = io();
let _rosOk = false, _procOk = false;
let _currentLogTab = 'log_main';
let _logs = { log_main: [], log_coords: [], log_collect: [] };
let _imgTs = { img_detect: 0, img_depth: 0 };

socket.on('connect', () => {
  document.getElementById('dot-net').classList.add('on');
  document.getElementById('s-net').style.background = 'var(--green)';
  document.getElementById('sys-status').textContent = 'CONNECTED';
});
socket.on('disconnect', () => {
  document.getElementById('dot-net').classList.remove('on');
  document.getElementById('sys-status').textContent = 'DISCONNECTED';
});

socket.on('status', d => {
  if (d.ros  !== undefined) setDot('ros',  d.ros);
  if (d.proc !== undefined) setDot('proc', d.proc);
  if (d.started !== undefined)
    document.getElementById('sys-status').textContent = d.started ? 'STARTED' : 'STOPPED';
});

socket.on('log', d => {
  _logs[d.channel] = _logs[d.channel] || [];
  _logs[d.channel].push(d.line);
  if (_logs[d.channel].length > 120) _logs[d.channel].shift();
  if (d.channel === _currentLogTab) appendLogLine(d.channel, d.line);
});

socket.on('clear_logs', () => {
  _logs = { log_main: [], log_coords: [], log_collect: [] };
  document.getElementById('log-view').innerHTML = '';
});

socket.on('image', d => {
  const now = Date.now();
  const key  = d.channel;           // 'img_detect' or 'img_depth'
  const id   = key === 'img_detect' ? 'img-detect' : 'img-depth';
  const ph   = key === 'img_detect' ? 'ph-detect'  : 'ph-depth';
  const fps  = key === 'img_detect' ? 'fps-detect' : 'fps-depth';
  const img  = document.getElementById(id);
  img.src = 'data:image/jpeg;base64,' + d.data;
  img.style.display = 'block';
  document.getElementById(ph).style.display = 'none';
  const dt = now - (_imgTs[key] || now);
  if (dt > 0) document.getElementById(fps).textContent = (1000/dt).toFixed(1) + ' fps';
  _imgTs[key] = now;
});

function setDot(name, on) {
  ['dot-'+name, 's-'+name].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('on', on);
    el.classList.toggle('err', !on);
    el.style.background = on ? 'var(--green)' : 'var(--red)';
  });
  if (name === 'ros') { _rosOk = on; document.getElementById('s-pub').style.background = on ? 'var(--green)' : 'var(--dim)'; }
  if (name === 'proc') _procOk = on;
}

function sendCmd(cmd, extra) {
  socket.emit('command', Object.assign({ cmd }, extra || {}));
}

// ── Tabs ────────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  const tabs = document.querySelectorAll('.tab');
  const map  = ['ctrl','cam','log','setup','depth'];
  tabs[map.indexOf(name)]?.classList.add('active');
  if (name === 'setup') loadSetup();
  if (name === 'depth') loadDepth();
  if (name === 'log')   renderLog();
  socket.emit('view', { tab: name });
}

// ── Logs ────────────────────────────────────────────────────────────────────
function switchLogTab(ch) {
  _currentLogTab = ch;
  document.querySelectorAll('.log-tab').forEach((t,i) => {
    t.classList.toggle('active', ['log_main','log_coords','log_collect'][i] === ch);
  });
  renderLog();
}
function renderLog() {
  const v = document.getElementById('log-view');
  v.innerHTML = '';
  (_logs[_currentLogTab] || []).forEach(l => appendLogLine(_currentLogTab, l, false));
  v.scrollTop = v.scrollHeight;
}
function appendLogLine(ch, line, scroll=true) {
  if (_currentLogTab !== ch) return;
  const v   = document.getElementById('log-view');
  const div = document.createElement('div');
  div.className = 'log-line' + (ch==='log_coords' ? ' coord' : ch==='log_collect' ? ' collect' : '');
  div.textContent = line;
  v.appendChild(div);
  if (scroll) v.scrollTop = v.scrollHeight;
}

// ── Setup ───────────────────────────────────────────────────────────────────
let _setupData = {}, _setupLabels = {};
let _depthData = {}, _depthLabels = {}, _depthMethod = 'bm';
let _activeEditor = null; // 'setup' or 'depth'

function loadSetup() {
  fetch('/api/config/setup').then(r=>r.json()).then(d => {
    _setupData   = d.data   || {};
    _setupLabels = d.labels || {};
    renderParamList('setup-rows', _setupData, _setupLabels, 'setup');
  });
}
function loadDepth() {
  fetch('/api/config/depth').then(r=>r.json()).then(d => {
    _depthData   = d.data   || {};
    _depthLabels = d.labels || {};
    _depthMethod = d.method || 'bm';
    document.getElementById('btn-bm').classList.toggle('active',   _depthMethod==='bm');
    document.getElementById('btn-sgbm').classList.toggle('active', _depthMethod==='sgbm');
    renderParamList('depth-rows', _depthData, _depthLabels, 'depth');
  });
}
function setMethod(m) {
  _depthMethod = m;
  fetch('/api/config/depth', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ stereo_method: m })
  }).then(() => loadDepth());
}

function renderParamList(containerId, data, labels, ctx) {
  const c = document.getElementById(containerId);
  c.innerHTML = '';
  let idx = 0;
  for (const [key, label] of Object.entries(labels)) {
    idx++;
    const val  = data[key];
    const type = typeof val === 'boolean' ? 'BOOL' : Array.isArray(val) ? 'LIST' : 'NUM';
    const tcol = type==='BOOL' ? '#ffd166' : type==='LIST' ? '#39ff7e' : '#00e5ff';
    const vcol = type==='BOOL' ? 'var(--yellow)' : type==='LIST' ? 'var(--green)' : 'var(--accent)';
    const vstr = fmtVal(val);

    const row = document.createElement('div');
    row.className = 'param-row';
    row.dataset.key = key;
    row.innerHTML = `
      <div class="accent-bar"></div>
      <div class="inner">
        <div class="top-line">
          <span class="idx">${String(idx).padStart(2,'0')}</span>
          <span class="lbl">${label}</span>
          <span class="badge" style="background:${tcol}">${type}</span>
        </div>
        <div class="bot-line">
          <span class="key-name">${key}</span>
          <span class="val" style="color:${vcol}" id="val-${ctx}-${key}">${vstr}</span>
        </div>
      </div>`;
    row.addEventListener('click', () => openEditor(ctx, key));
    c.appendChild(row);
  }
}

function fmtVal(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Array.isArray(v)) {
    const s = '[' + v.map(x=>typeof x==='number'&&!Number.isInteger(x)?x.toFixed(3):String(x)).join(', ') + ']';
    return s.length > 32 ? s.slice(0,29)+'…]' : s;
  }
  return typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : String(v);
}
function fmtValFull(v) {
  if (Array.isArray(v)) return '['+v.map(x=>typeof x==='number'&&!Number.isInteger(x)?x.toFixed(3):String(x)).join(', ')+']';
  return fmtVal(v);
}

function openEditor(ctx, key) {
  _activeEditor = ctx;
  const data   = ctx==='setup' ? _setupData   : _depthData;
  const labels = ctx==='setup' ? _setupLabels : _depthLabels;
  const val    = data[key];
  const label  = labels[key] || key;
  const isBool = typeof val === 'boolean';
  const isList = Array.isArray(val);

  // mark selected row
  document.querySelectorAll(`#${ctx==='setup'?'setup':'depth'}-rows .param-row`).forEach(r => {
    r.classList.toggle('sel', r.dataset.key === key);
    r.querySelector('.accent-bar').style.background = r.dataset.key===key ? 'var(--accent)' : 'var(--card)';
  });

  const formId  = ctx==='setup' ? 'editor-form'       : 'depth-editor-form';
  const vkbdId  = ctx==='setup' ? 'vkbd'              : 'depth-vkbd';
  const listId  = ctx==='setup' ? 'setup-list'        : 'depth-list';
  const editorId= ctx==='setup' ? 'editor'            : 'depth-editor';

  document.getElementById(listId).style.display  = 'none';
  const ed = document.getElementById(editorId);
  ed.style.display = 'flex';
  ed.classList.add('active');

  const form = document.getElementById(formId);
  form.innerHTML = '';

  // Back
  const back = document.createElement('button');
  back.className = 'back-btn';
  back.innerHTML = '← BACK';
  back.onclick = () => {
    ed.style.display = 'none';
    ed.classList.remove('active');
    document.getElementById(listId).style.display = 'flex';
    document.getElementById(vkbdId).style.display = 'none';
  };
  form.appendChild(back);

  // Title
  const tr = document.createElement('div');
  tr.innerHTML = `<span class="editor-title">${label}</span><span class="editor-key">· ${key}</span>`;
  form.appendChild(tr);

  // Pill
  const pill = document.createElement('div');
  pill.className = 'curr-pill';
  pill.innerHTML = `<div class="cl">CURRENT VALUE</div><div class="cv">${fmtValFull(val)}</div>`;
  form.appendChild(pill);

  const sep = document.createElement('div');
  sep.style.cssText = 'height:1px;background:var(--border);margin-bottom:14px';
  form.appendChild(sep);

  if (isBool) {
    // Bool toggle
    let cur = val;
    const lbl = document.createElement('div'); lbl.className='input-lbl'; lbl.textContent='Toggle value:';
    form.appendChild(lbl);
    const br = document.createElement('div'); br.className='bool-row';
    const bon = document.createElement('button'); bon.className='bool-btn on'+(cur?' sel-on':''); bon.textContent='● ON';
    const bof = document.createElement('button'); bof.className='bool-btn off'+(!cur?' sel-off':''); bof.textContent='● OFF';
    bon.onclick = () => { cur=true;  bon.className='bool-btn on sel-on'; bof.className='bool-btn off'; };
    bof.onclick = () => { cur=false; bof.className='bool-btn off sel-off'; bon.className='bool-btn on'; };
    br.append(bon, bof); form.appendChild(br);
    const sv = document.createElement('button'); sv.className='save-btn'; sv.textContent='💾  SAVE CHANGES';
    sv.onclick = () => saveParam(ctx, key, cur);
    form.appendChild(sv);

  } else {
    // Numeric / list input
    const lbl = document.createElement('div'); lbl.className='input-lbl';
    lbl.textContent = isList ? 'Edit list (comma-separated):' : 'New value:';
    form.appendChild(lbl);
    const wrap = document.createElement('div'); wrap.className='input-wrap';
    const inp  = document.createElement('input'); inp.type='text';
    inp.value  = isList ? val.join(', ') : String(val);
    inp.setAttribute('readonly', true);
    inp.addEventListener('focus', () => inp.removeAttribute('readonly'));
    wrap.appendChild(inp); form.appendChild(wrap);
    const sv = document.createElement('button'); sv.className='save-btn'; sv.textContent='💾  SAVE CHANGES';
    sv.onclick = () => {
      let nv;
      if (isList) {
        try { nv = inp.value.split(',').map(p => { const t=p.trim(); return t.includes('.')?parseFloat(t):parseInt(t,10); }); }
        catch(e) { toast('Invalid list format'); return; }
      } else {
        nv = typeof val==='number'&&Number.isInteger(val) ? parseInt(inp.value,10) : parseFloat(inp.value);
        if (isNaN(nv)) { toast('Invalid number'); return; }
      }
      saveParam(ctx, key, nv);
    };
    form.appendChild(sv);
    // show keyboard
    buildVKbd(vkbdId, inp);
    document.getElementById(vkbdId).style.display = 'block';
  }
}

function saveParam(ctx, key, val) {
  const body = { [key]: val };
  if (ctx === 'depth') body.stereo_method = _depthMethod;
  fetch(`/api/config/${ctx}`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  }).then(r=>r.json()).then(d => {
    if (d.ok) {
      if (ctx==='setup') _setupData[key]=val; else _depthData[key]=val;
      const vEl = document.getElementById(`val-${ctx}-${key}`);
      if (vEl) vEl.textContent = fmtVal(val);
      toast('✓  Saved');
    } else toast('Error: ' + d.error);
  });
}

// ── Virtual keyboard ─────────────────────────────────────────────────────────
function buildVKbd(containerId, target) {
  const rows = ['`1234567890-=','qwertyuiop[]\\','asdfghjkl;\'','zxcvbnm,./'];
  const shift_map = {'`':'~','1':'!','2':'@','3':'#','4':'$','5':'%','6':'^','7':'&','8':'*','9':'(','0':')','-':'_','=':'+','[':'{',']':'}','\\':'|',';':':','\'':'"',',':'<','.':'>','/':'?'};
  let shift=false, caps=false;
  const c = document.getElementById(containerId);
  c.innerHTML = '';

  function type(ch) {
    const active = shift||caps;
    const out = active ? (shift_map[ch]||ch.toUpperCase()) : ch;
    const pos = target.selectionStart;
    const v = target.value;
    target.value = v.slice(0,pos) + out + v.slice(target.selectionEnd);
    target.setSelectionRange(pos+1,pos+1);
    target.focus();
    if (shift&&!caps){ shift=false; updateKeys(); }
  }
  function backspace() {
    const pos = target.selectionStart;
    if (pos===0) return;
    const v = target.value;
    target.value = v.slice(0,pos-1)+v.slice(pos);
    target.setSelectionRange(pos-1,pos-1);
    target.focus();
  }
  const allBtns = [];
  function updateKeys() {
    const active = shift||caps;
    allBtns.forEach(([ch,b]) => { if(ch.length===1) b.textContent = active?(shift_map[ch]||ch.toUpperCase()):ch; });
  }

  rows.forEach(row => {
    const rf = document.createElement('div'); rf.className='kbd-row';
    [...row].forEach(ch => {
      const b = document.createElement('button'); b.className='kkey'; b.textContent=ch;
      b.addEventListener('click', ()=>type(ch));
      rf.appendChild(b); allBtns.push([ch,b]);
    });
    c.appendChild(rf);
  });

  const sp = document.createElement('div'); sp.className='kbd-row';
  const specials = [
    ['SHIFT',()=>{shift=!shift;updateKeys();},'sp'],
    ['CAPS', ()=>{caps=!caps;updateKeys();},'sp'],
    ['SPACE',()=>{type(' ');},'sp space-key'],
    ['⌫',  backspace,'sp'],
    ['CLR', ()=>{target.value='';target.focus();},'sp'],
  ];
  specials.forEach(([t,fn,cls])=>{
    const b = document.createElement('button'); b.className='kkey '+cls; b.textContent=t;
    b.addEventListener('click', fn); sp.appendChild(b);
  });
  c.appendChild(sp);
}

// ── Toast ────────────────────────────────────────────────────────────────────
let _toastTimer;
function toast(msg, dur=1800) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.display='block';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(()=>{ t.style.display='none'; }, dur);
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetch('/api/logs').then(r=>r.json()).then(d => {
    for (const ch of Object.keys(d)) _logs[ch] = d[ch] || [];
  });
  // Auto-load QR on first open
  _loadQR();
});

// ── QR Code ─────────────────────────────────────────────────────────────────
let _qrLoaded = false;
function _loadQR() {
  if (_qrLoaded) return;
  fetch('/api/qr').then(r=>r.json()).then(d => {
    if (!d.qr) return;
    document.getElementById('qr-img').src = 'data:image/png;base64,' + d.qr;
    const badge = d.is_public ? '🌐 PUBLIC — mọi nơi' : '📶 LOCAL — cùng WiFi';
    document.getElementById('qr-url').innerHTML =
      `<span style="font-size:11px;color:${d.is_public?'#39ff7e':'#ffd166'}">${badge}</span><br>${d.url}`;
    _qrLoaded = true;
  }).catch(()=>{});
}
function showQR() {
  _qrLoaded = false;   // reload mỗi lần mở để lấy ngrok URL mới nhất
  _loadQR();
  document.getElementById('qr-overlay').classList.add('show');
}
function closeQR() {
  document.getElementById('qr-overlay').classList.remove('show');
}
</script>
</body>
</html>
"""

# ── IP detection ─────────────────────────────────────────────────────────────
def _get_all_ips():
    """Lấy tất cả IP LAN của máy, bỏ qua loopback."""
    import socket as _sock
    ips = []

    # Method 1: UDP trick → IP ra internet/gateway chính
    for target in ("8.8.8.8", "192.168.1.1", "10.0.0.1"):
        try:
            s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
            s.settimeout(0.3)
            s.connect((target, 80))
            addr = s.getsockname()[0]
            s.close()
            if addr not in ips and not addr.startswith("127."):
                ips.append(addr)
        except:
            pass

    # Method 2: getaddrinfo từ hostname
    try:
        for info in _sock.getaddrinfo(_sock.gethostname(), None, _sock.AF_INET):
            addr = info[4][0]
            if addr not in ips and not addr.startswith("127."):
                ips.append(addr)
    except:
        pass

    return ips if ips else ["0.0.0.0"]


# ── Health check endpoint ─────────────────────────────────────────────────────
@app.route('/ping')
def ping():
    """Test nhanh từ điện thoại: mở http://IP:5000/ping — nếu thấy JSON là OK."""
    return jsonify({
        "ok":     True,
        "server": "INDY ROS2 HMI",
        "ros":    _state["ros_ok"],
        "time":   _ts(),
        "ips":    _get_all_ips(),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
_shutdown_flag = threading.Event()

def _handle_sigint(sig, frame):
    print("\n[SERVER] Ctrl+C — shutting down...")
    _shutdown_flag.set()
    # Tắt ROS
    global _ros_worker, _ros_node
    if _ros_worker:
        try: _ros_worker.stop()
        except: pass
    if _ros_node:
        try:
            import rclpy
            _ros_node.destroy_node()
            rclpy.shutdown()
        except: pass
    # Tắt process con nếu có
    global _proc, _proc_running
    _proc_running = False
    if _proc:
        try: os.killpg(os.getpgid(_proc.pid), _signal.SIGTERM)
        except: pass
    os._exit(0)   # thoát ngay, không chờ thread


def _get_qr_base64(url: str) -> str:
    """Tạo QR code PNG dưới dạng base64 string."""
    try:
        import qrcode, io, base64
        qr = qrcode.QRCode(
            version=3,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6, border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#00e5ff", back_color="#0d0f14")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[SERVER] QR gen failed: {e}")
        return ""


@app.route('/api/qr')
def api_qr():
    """Trả về QR code — ưu tiên ngrok URL nếu có, fallback local IP."""
    # Đợi tối đa 8s cho ngrok khởi động
    for _ in range(16):
        if _state.get("ngrok_url") is not None or _state.get("qr_url"):
            break
        time.sleep(0.5)

    url = _state.get("ngrok_url") or _state.get("qr_url", "http://localhost:5000")
    b64 = _get_qr_base64(url)
    return jsonify({
        "url":      url,
        "qr":       b64,
        "ips":      _get_all_ips(),
        "ngrok":    _state.get("ngrok_url"),
        "is_public": bool(_state.get("ngrok_url")),
    })


def main():
    # Signal chỉ đăng ký được trên main thread
    import threading as _th
    if _th.current_thread() is _th.main_thread():
        _signal.signal(_signal.SIGINT,  _handle_sigint)
        _signal.signal(_signal.SIGTERM, _handle_sigint)

    ips  = _get_all_ips()
    port = 5000

    threading.Thread(target=_init_ros, daemon=True).start()

    # ── Khởi ngrok tunnel trong thread riêng ──────────────────────────────────
    def _start_ngrok():
        try:
            from pyngrok import ngrok, conf
            # Nếu có auth token thì set ở đây (miễn phí tại ngrok.com):
            conf.get_default().auth_token = YOUR_NGROK_TOKEN
            tunnel     = ngrok.connect(port, "http")
            public_url = tunnel.public_url
            _state["ngrok_url"] = public_url

            print(f"\n{'='*58}")
            print(f"  INDY ROS2 WEB HMI — NGROK TUNNEL READY")
            print(f"{'='*58}")
            print(f"  Public URL (mọi nơi, không cần cùng WiFi):")
            print(f"    {public_url}")
            print(f"\n  Local URL (cùng WiFi):")
            for ip in ips:
                print(f"    http://{ip}:{port}")
            print(f"{'='*58}\n")

            # Cập nhật QR về public URL
            _state["qr_url"] = public_url

        except ImportError:
            print("[SERVER] pyngrok not installed — chỉ dùng local URL")
            print(f"[SERVER] pip install pyngrok")
            _state["ngrok_url"] = None
            _print_local_urls(ips, port)
        except Exception as e:
            print(f"[SERVER] ngrok error: {e}")
            _state["ngrok_url"] = None
            _print_local_urls(ips, port)

    def _print_local_urls(ips, port):
        print(f"\n{'='*58}")
        print(f"  INDY ROS2 WEB HMI")
        print(f"{'='*58}")
        print(f"  Local URL (cùng WiFi):")
        for ip in ips:
            print(f"    http://{ip}:{port}")
        print(f"{'='*58}\n")

    # Khởi ngrok trước (non-blocking)
    _state["ngrok_url"] = None
    _state["qr_url"]    = f"http://{ips[0]}:{port}" if ips else f"http://localhost:{port}"
    threading.Thread(target=_start_ngrok, daemon=True).start()

    # Start log emitter thread (batch WebSocket emit)
    threading.Thread(target=_log_emitter, daemon=True, name="log-emitter").start()

    sio.run(app, host='0.0.0.0', port=port,
            debug=False, use_reloader=False,
            allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
