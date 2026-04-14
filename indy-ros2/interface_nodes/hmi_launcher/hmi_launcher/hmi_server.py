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
    "img_detect": None, "img_depth": None,
}
_ros_node    = None
_ros_worker  = None
_start_pub   = None
_proc        = None
_proc_running = False

# ── Helpers ───────────────────────────────────────────────────────────────────
def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _push_log(channel, line):
    """Ghi log qua bus (bus sẽ notify cả Tkinter HMI lẫn WebSocket)."""
    bus.push(channel, line)


def _bus_to_ws(channel, line):
    """Callback đăng ký với bus — forward mọi log mới ra tất cả WebSocket client."""
    if channel == '__clear__':
        try:
            sio.emit('clear_logs', {})
        except Exception:
            pass
        return
    try:
        sio.emit('log', {'channel': channel, 'line': line})
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
            sio.emit('image', {'channel': key, 'data': b64})
        except Exception as e:
            pass

def _init_ros():
    global _ros_node, _ros_worker, _start_pub
    if not HAS_ROS:
        print('[SERVER] rclpy not found — ROS disabled')
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
        print('[SERVER] ROS2 node ready')
        sio.emit('status', {'ros': True})
    except Exception as e:
        print(f'[SERVER] ROS2 init error: {e}')

# ── Process (AUTO_CMD) ────────────────────────────────────────────────────────
def _read_proc(proc):
    global _proc_running
    try:
        while _proc_running and proc:
            line = proc.stdout.readline()
            if not line: break
            line = line.strip()
            if not line: continue
            if len(line) > 400: line = line[:400] + '...'
            bus.classify_and_push(line, COORD_KW, COLLECT_KW)
    except: pass
    finally:
        _state['proc_ok'] = False
        try:
            sio.emit('status', {'proc': False})
        except Exception:
            pass

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
        print(f'[SETUP] Saved {list(body.keys())}')
    elif name == 'depth':
        raw  = _load_yaml(DEPTH_PATH)
        meth = body.pop('stereo_method', raw.get('stereo_method', 'bm'))
        raw['stereo_method'] = meth
        raw.setdefault(f'stereo_{meth}', {}).update(body)
        err = _save_yaml(DEPTH_PATH, raw)
        print(f'[DEPTH] Saved method={meth} {list(body.keys())}')
    else:
        return jsonify({'ok': False, 'error': 'unknown'}), 404
    return jsonify({'ok': err is True, 'error': err if err is not True else None})

@app.route('/api/logs')
def api_logs():
    return jsonify(bus.snapshot())

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
    # send cached logs from bus
    snap = bus.snapshot()
    for ch in ('log_main', 'log_coords', 'log_collect'):
        for line in snap.get(ch, [])[-50:]:
            emit('log', {'channel': ch, 'line': line})
    # send cached images
    for k in ('img_detect', 'img_depth'):
        if _state.get(k):
            emit('image', {'channel': k, 'data': _state[k]})

@sio.on('command')
def on_command(data):
    global _proc, _proc_running
    cmd = data.get('cmd')

    if cmd == 'start':
        if HAS_ROS and _start_pub:
            try:
                msg = StartMsg(); msg.start = True
                _start_pub.publish(msg)
                print('[HMI] Published start=True')
                emit('status', {'started': True})
            except Exception as e:
                print(f'[HMI] Publish error: {e}')
        else:
            print('[HMI] ROS unavailable')

    elif cmd == 'stop':
        if HAS_ROS and _start_pub:
            try:
                msg = StartMsg(); msg.start = False
                _start_pub.publish(msg)
                print('[HMI] Published start=False')
                emit('status', {'started': False})
            except Exception as e:
                print(f'[HMI] Publish error: {e}')

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
            print(f'[SERVER] Launched PID {_proc.pid}')
            sio.emit('status', {'proc': True})
        except Exception as e:
            print(f'[SERVER] Launch error: {e}')

    elif cmd == 'kill':
        _proc_running = False
        if _proc:
            try:
                os.killpg(os.getpgid(_proc.pid), _signal.SIGTERM)
                print('[SERVER] Process killed')
            except: pass
        _state['proc_ok'] = False
        sio.emit('status', {'proc': False})

    elif cmd == 'clear_logs':
        bus.clear()   # clears all channels and notifies subscribers (incl. _bus_to_ws)

# ── HTML App (single-file PWA) ────────────────────────────────────────────────
HTML_APP = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0b0e15">
<title>INDY ROS2 HMI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
:root {
  --bg:       #0b0e15;
  --surface:  #111520;
  --card:     #161c2a;
  --card-h:   #1c2538;
  --border:   #1e2840;
  --accent:   #00e5ff;
  --green:    #39ff7e;
  --red:      #ff3b5c;
  --yellow:   #ffd166;
  --text:     #c8d6e5;
  --dim:      #4a5568;
  --sub:      #7a8fa8;
  --font:     'Courier New', monospace;
  --radius:   10px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); overflow: hidden; }
 
/* ── Root layout ── */
#app { display: flex; flex-direction: column; height: 100vh; }
 
/* ── Top bar ── */
#topbar {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--surface); height: 54px; padding: 0 18px; flex-shrink: 0;
  border-bottom: 2px solid var(--accent);
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand .hex { color: var(--accent); font-size: 22px; }
.brand .title { color: var(--accent); font-weight: bold; font-size: 15px; letter-spacing: .05em; }
.brand .st { color: var(--dim); font-size: 11px; margin-left: 4px; }
.top-right { display: flex; align-items: center; gap: 10px; }
.dot-row { display: flex; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); transition: background .3s; }
.dot.on  { background: var(--green); }
.dot.err { background: var(--red); }
.qr-btn {
  background: none; border: 1px solid var(--border); color: var(--accent);
  font-family: var(--font); font-size: 11px; font-weight: bold;
  padding: 5px 12px; border-radius: 6px; cursor: pointer;
}
.qr-btn:active { opacity: .7; }
 
/* ── Screen system ── */
.screen { display: none; flex: 1; flex-direction: column; overflow: hidden; }
.screen.active { display: flex; }
 
/* ══════════════════════ HOME SCREEN ══════════════════════ */
#home { padding: 0; overflow-y: auto; }
.home-body { padding: 20px 18px; display: flex; flex-direction: column; gap: 14px; }
 
/* Status bar */
.status-bar {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
}
.status-item {
  background: var(--card); border-radius: var(--radius); padding: 12px 14px;
  display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--sub);
  border: 1px solid var(--border);
}
.status-item .dot { flex-shrink: 0; width: 10px; height: 10px; }
 
/* Big control buttons */
.ctrl-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.ctrl-btn {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 8px; padding: 22px 12px; border: none; border-radius: var(--radius);
  font-family: var(--font); font-size: 13px; font-weight: bold; letter-spacing: .05em;
  cursor: pointer; transition: all .15s; border: 1px solid transparent;
}
.ctrl-btn:active { transform: scale(.96); opacity: .85; }
.ctrl-btn .icon { font-size: 26px; }
.ctrl-btn.start  { background: #0d2416; color: var(--green);  border-color: #1a4a2a; }
.ctrl-btn.stop   { background: #2a0e0e; color: var(--red);    border-color: #4a1a1a; }
.ctrl-btn.log    { background: #0e1a2a; color: var(--accent); border-color: #1a2e4a; }
.ctrl-btn.cam    { background: #1a1a0e; color: var(--yellow); border-color: #3a3a1a; }
.ctrl-btn.setup  { background: #1a0e2a; color: #c084fc;       border-color: #3a1a4a; }
 
/* Section label */
.sec-label { font-size: 10px; font-weight: bold; color: var(--dim); letter-spacing: .12em; text-transform: uppercase; padding: 4px 2px; }
 
/* ══════════════════════ INNER SCREEN (Log / Camera / Settings) ══════════════════════ */
.inner-header {
  display: flex; align-items: center; gap: 12px;
  background: var(--surface); height: 52px; padding: 0 16px; flex-shrink: 0;
  border-bottom: 1px solid var(--border);
}
.back-btn {
  background: var(--card); border: 1px solid var(--border); color: var(--text);
  font-family: var(--font); font-size: 13px; font-weight: bold;
  padding: 7px 14px; border-radius: 6px; cursor: pointer; flex-shrink: 0;
}
.back-btn:active { opacity: .7; }
.inner-title { color: var(--accent); font-size: 14px; font-weight: bold; flex: 1; }
 
/* ── LOG SCREEN ── */
#log-screen .log-body { display: flex; flex: 1; overflow: hidden; }
 
/* Left sidebar: log type selector */
.log-sidebar {
  width: 110px; flex-shrink: 0; background: var(--surface);
  border-right: 1px solid var(--border); display: flex; flex-direction: column;
  padding: 10px 8px; gap: 6px;
}
.log-tab-btn {
  background: var(--card); border: 1px solid var(--border);
  color: var(--sub); font-family: var(--font); font-size: 11px; font-weight: bold;
  padding: 12px 8px; border-radius: 8px; cursor: pointer; text-align: center;
  transition: all .15s; line-height: 1.4;
}
.log-tab-btn.active { background: var(--card-h); color: var(--accent); border-color: var(--accent); }
.log-tab-btn:active { opacity: .7; }
.log-sidebar-footer { margin-top: auto; padding-top: 8px; }
.clear-btn {
  width: 100%; background: #2a0e0e; border: 1px solid #4a1a1a;
  color: var(--red); font-family: var(--font); font-size: 11px; font-weight: bold;
  padding: 10px 6px; border-radius: 8px; cursor: pointer; text-align: center;
}
.clear-btn:active { opacity: .7; }
 
/* Right: log content */
.log-panel { flex: 1; overflow-y: auto; padding: 10px 12px; font-size: 11px; line-height: 1.7; }
.log-line { padding: 2px 0; border-bottom: 1px solid rgba(30,40,64,.35); word-break: break-all; }
.log-line.coord   { color: var(--green); }
.log-line.collect { color: var(--yellow); }
 
/* ── CAMERA SCREEN ── */
#cam-screen .cam-body { display: flex; flex-direction: column; flex: 1; gap: 4px; padding: 4px; overflow: hidden; }
.cam-box {
  flex: 1; background: #050608; border-radius: 8px;
  position: relative; overflow: hidden;
  display: flex; align-items: center; justify-content: center;
}
.cam-box img { width: 100%; height: 100%; object-fit: contain; display: block; }
.cam-label {
  position: absolute; top: 8px; left: 10px;
  font-size: 10px; font-weight: bold; color: var(--accent);
  background: rgba(0,0,0,.65); padding: 3px 8px; border-radius: 4px;
}
.cam-fps {
  position: absolute; top: 8px; right: 10px;
  font-size: 10px; color: var(--dim);
  background: rgba(0,0,0,.65); padding: 3px 8px; border-radius: 4px;
}
.cam-ph { color: var(--dim); font-size: 12px; }
 
/* ── SETTINGS SCREEN ── */
#settings-screen .settings-body { display: flex; flex: 1; overflow: hidden; }
 
/* Left sidebar: settings sections */
.set-sidebar {
  width: 110px; flex-shrink: 0; background: var(--surface);
  border-right: 1px solid var(--border); display: flex; flex-direction: column;
  padding: 10px 8px; gap: 6px;
}
.set-tab-btn {
  background: var(--card); border: 1px solid var(--border);
  color: var(--sub); font-family: var(--font); font-size: 11px; font-weight: bold;
  padding: 12px 8px; border-radius: 8px; cursor: pointer; text-align: center;
  transition: all .15s; line-height: 1.4;
}
.set-tab-btn.active { background: var(--card-h); color: var(--accent); border-color: var(--accent); }
 
/* Right: param list or editor */
.set-panel { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
 
/* Param list */
.param-list { overflow-y: auto; flex: 1; }
.param-row {
  display: flex; align-items: stretch; background: var(--card);
  border-bottom: 1px solid var(--border); cursor: pointer; min-height: 62px;
  transition: background .12s;
}
.param-row:active, .param-row.sel { background: var(--card-h); }
.param-row .accent-bar { width: 3px; background: transparent; flex-shrink: 0; transition: background .15s; }
.param-row.sel .accent-bar { background: var(--accent); }
.param-row .inner { flex: 1; padding: 11px 14px; min-width: 0; }
.param-row .top-line { display: flex; align-items: center; gap: 6px; }
.param-row .idx { color: var(--dim); font-size: 10px; flex-shrink: 0; }
.param-row .lbl { flex: 1; font-size: 13px; font-weight: bold; color: var(--text); }
.badge { font-size: 9px; font-weight: bold; color: #0b0e15; padding: 2px 6px; border-radius: 3px; flex-shrink: 0; }
.param-row .bot-line { display: flex; justify-content: space-between; margin-top: 4px; }
.param-row .key-name { font-size: 10px; color: var(--dim); }
.param-row .val { font-size: 12px; font-weight: bold; }
 
/* Param editor */
.param-editor { flex: 1; overflow-y: auto; padding: 16px; display: none; flex-direction: column; }
.param-editor.active { display: flex; }
.editor-back { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; }
.editor-title { color: var(--accent); font-size: 15px; font-weight: bold; }
.editor-key   { color: var(--dim); font-size: 11px; }
.curr-pill {
  background: var(--card); border-left: 3px solid var(--accent);
  border-radius: 6px; padding: 12px 14px; margin-bottom: 14px;
}
.curr-pill .cl { color: var(--dim); font-size: 9px; font-weight: bold; letter-spacing: .1em; margin-bottom: 4px; }
.curr-pill .cv { color: var(--yellow); font-size: 13px; font-weight: bold; word-break: break-all; }
.inp-label { color: var(--sub); font-size: 11px; margin-bottom: 8px; }
.inp-wrap { background: var(--accent); padding: 2px; border-radius: 6px; margin-bottom: 14px; }
.inp-wrap input {
  width: 100%; background: var(--card); color: var(--accent);
  font-family: var(--font); font-size: 16px; font-weight: bold;
  border: none; padding: 12px 14px; outline: none; border-radius: 5px;
}
.save-btn {
  background: var(--green); color: #0b0e15; font-family: var(--font);
  font-size: 13px; font-weight: bold; border: none; border-radius: 8px;
  padding: 14px; cursor: pointer; width: 100%; margin-top: 4px;
}
.save-btn:active { opacity: .8; }
.bool-row { display: flex; gap: 10px; margin-bottom: 14px; }
.bool-btn {
  flex: 1; padding: 16px; font-family: var(--font); font-size: 14px; font-weight: bold;
  border: 2px solid var(--border); border-radius: 8px; cursor: pointer; background: var(--card); color: var(--sub);
  transition: all .15s;
}
.bool-btn.sel-on  { border-color: var(--green); color: var(--green); background: #0d2416; }
.bool-btn.sel-off { border-color: var(--red);   color: var(--red);   background: #2a0e0e; }
 
/* Method toggle */
.method-bar { display: flex; gap: 8px; padding: 10px 12px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }
.method-btn {
  flex: 1; background: var(--card); border: 1px solid var(--border);
  color: var(--sub); font-family: var(--font); font-size: 12px; font-weight: bold;
  padding: 10px; border-radius: 6px; cursor: pointer; text-align: center; transition: all .15s;
}
.method-btn.active { background: var(--accent); color: #0b0e15; border-color: var(--accent); }
 
/* ── QR overlay ── */
#qr-overlay {
  display: none; position: fixed; inset: 0; z-index: 9999;
  background: rgba(0,0,0,.88); flex-direction: column;
  align-items: center; justify-content: center; gap: 16px;
}
#qr-overlay.show { display: flex; }
#qr-overlay img { width: 230px; height: 230px; border-radius: 10px; }
.qr-info { text-align: center; }
.qr-badge { font-size: 11px; font-weight: bold; margin-bottom: 6px; }
.qr-url-text { color: var(--accent); font-size: 13px; font-weight: bold; font-family: var(--font); }
.qr-close-btn {
  background: var(--card); border: 1px solid var(--border); color: var(--text);
  font-family: var(--font); font-size: 13px; font-weight: bold;
  padding: 12px 32px; border-radius: 8px; cursor: pointer; margin-top: 6px;
}
 
/* ── Toast ── */
#toast {
  position: fixed; bottom: 28px; left: 50%; transform: translateX(-50%);
  background: var(--card-h); color: var(--text); font-size: 12px; font-weight: bold;
  padding: 10px 22px; border-radius: 20px; border: 1px solid var(--border);
  display: none; z-index: 9998; white-space: nowrap; font-family: var(--font);
}
</style>
</head>
<body>
 
<!-- QR Overlay -->
<div id="qr-overlay" onclick="closeQR()">
  <img id="qr-img" src="" alt="QR">
  <div class="qr-info">
    <div class="qr-badge" id="qr-badge"></div>
    <div class="qr-url-text" id="qr-url-text"></div>
  </div>
  <button class="qr-close-btn">✕  CLOSE</button>
</div>
 
<div id="app">
 
  <!-- Top bar -->
  <div id="topbar">
    <div class="brand">
      <span class="hex">⬡</span>
      <span class="title">INDY ROS2</span>
      <span class="st" id="sys-status">CONNECTING…</span>
    </div>
    <div class="top-right">
      <button class="qr-btn" onclick="showQR()">QR</button>
      <div class="dot-row">
        <div class="dot" id="dot-ros"  title="ROS2"></div>
        <div class="dot" id="dot-proc" title="Process"></div>
        <div class="dot" id="dot-net"  title="Network"></div>
      </div>
    </div>
  </div>
 
  <!-- ════════ HOME ════════ -->
  <div class="screen active" id="home">
    <div class="home-body">
 
      <div class="sec-label">System Status</div>
      <div class="status-bar">
        <div class="status-item"><div class="dot" id="s-ros"></div>ROS2 Node</div>
        <div class="status-item"><div class="dot" id="s-proc"></div>Launch Proc</div>
        <div class="status-item"><div class="dot" id="s-net" style="background:var(--green)"></div>Network</div>
        <div class="status-item"><div class="dot" id="s-pub"></div>Publisher</div>
      </div>
 
      <div class="sec-label">Robot Control</div>
      <div class="ctrl-row">
        <button class="ctrl-btn start" onclick="sendCmd('start')">
          <span class="icon">▶</span>START
        </button>
        <button class="ctrl-btn stop" onclick="sendCmd('stop')">
          <span class="icon">■</span>STOP
        </button>
      </div>
 
      <div class="sec-label">Monitor</div>
      <div class="ctrl-row">
        <button class="ctrl-btn log" onclick="goScreen('log-screen')">
          <span class="icon">📋</span>LOG
        </button>
        <button class="ctrl-btn cam" onclick="goScreen('cam-screen')">
          <span class="icon">📷</span>CAMERA
        </button>
      </div>
 
      <div class="sec-label">Configuration</div>
      <button class="ctrl-btn setup" style="width:100%;flex-direction:row;gap:14px;padding:18px 20px;justify-content:flex-start"
              onclick="goScreen('settings-screen')">
        <span class="icon" style="font-size:22px">⚙</span>SETTINGS
      </button>
 
    </div>
  </div>
 
  <!-- ════════ LOG SCREEN ════════ -->
  <div class="screen" id="log-screen">
    <div class="inner-header">
      <button class="back-btn" onclick="goHome()">← Back</button>
      <span class="inner-title">Log Viewer</span>
    </div>
    <div class="log-body">
      <!-- Sidebar -->
      <div class="log-sidebar">
        <button class="log-tab-btn active" id="lt-main"    onclick="switchLog('log_main')">Main<br>Log</button>
        <button class="log-tab-btn"        id="lt-coords"  onclick="switchLog('log_coords')">Target<br>Coords</button>
        <button class="log-tab-btn"        id="lt-collect" onclick="switchLog('log_collect')">Collect<br>Log</button>
        <div class="log-sidebar-footer">
          <button class="clear-btn" onclick="sendCmd('clear_logs')">🗑 Clear</button>
        </div>
      </div>
      <!-- Log content -->
      <div class="log-panel" id="log-panel"></div>
    </div>
  </div>
 
  <!-- ════════ CAMERA SCREEN ════════ -->
  <div class="screen" id="cam-screen">
    <div class="inner-header">
      <button class="back-btn" onclick="goHome()">← Back</button>
      <span class="inner-title">Camera Feed</span>
    </div>
    <div class="cam-body">
      <div class="cam-box">
        <img id="img-detect" style="display:none">
        <div class="cam-ph" id="ph-detect">● no signal</div>
        <div class="cam-label">DETECT</div>
        <div class="cam-fps" id="fps-detect">–</div>
      </div>
      <div class="cam-box">
        <img id="img-depth" style="display:none">
        <div class="cam-ph" id="ph-depth">● no signal</div>
        <div class="cam-label">DEPTH</div>
        <div class="cam-fps" id="fps-depth">–</div>
      </div>
    </div>
  </div>
 
  <!-- ════════ SETTINGS SCREEN ════════ -->
  <div class="screen" id="settings-screen">
    <div class="inner-header">
      <button class="back-btn" onclick="goHome()">← Back</button>
      <span class="inner-title" id="set-title">Setup</span>
    </div>
    <div class="settings-body">
      <!-- Sidebar -->
      <div class="set-sidebar">
        <button class="set-tab-btn active" id="st-setup" onclick="switchSet('setup')">Robot<br>Setup</button>
        <button class="set-tab-btn"        id="st-depth" onclick="switchSet('depth')">Depth<br>Config</button>
      </div>
      <!-- Panel -->
      <div class="set-panel" id="set-panel">
        <!-- method bar (depth only) -->
        <div class="method-bar" id="method-bar" style="display:none">
          <button class="method-btn active" id="btn-bm"   onclick="setDepthMethod('bm')">BM</button>
          <button class="method-btn"        id="btn-sgbm" onclick="setDepthMethod('sgbm')">SGBM</button>
        </div>
        <!-- param list -->
        <div class="param-list" id="param-list"></div>
        <!-- param editor (shown on row click) -->
        <div class="param-editor" id="param-editor">
          <div class="editor-back">
            <button class="back-btn" onclick="closeEditor()">← Back</button>
            <div>
              <div class="editor-title" id="ed-title"></div>
              <div class="editor-key"   id="ed-key"></div>
            </div>
          </div>
          <div class="curr-pill">
            <div class="cl">CURRENT VALUE</div>
            <div class="cv" id="ed-curval"></div>
          </div>
          <div id="ed-input-area"></div>
        </div>
      </div>
    </div>
  </div>
 
</div>
<div id="toast"></div>
 
<script>
// ══════════════════ SOCKET ══════════════════
const socket = io();
let _logs = { log_main:[], log_coords:[], log_collect:[] };
let _curLog = 'log_main';
let _imgTs  = { img_detect:0, img_depth:0 };
 
socket.on('connect', () => {
  setDot('net', true);
  document.getElementById('sys-status').textContent = 'CONNECTED';
});
socket.on('disconnect', () => {
  setDot('net', false);
  document.getElementById('sys-status').textContent = 'DISCONNECTED';
});
socket.on('status', d => {
  if (d.ros   !== undefined) setDot('ros',  d.ros);
  if (d.proc  !== undefined) setDot('proc', d.proc);
  if (d.started !== undefined)
    document.getElementById('sys-status').textContent = d.started ? 'STARTED' : 'STOPPED';
});
socket.on('log', d => {
  _logs[d.channel] = _logs[d.channel] || [];
  _logs[d.channel].push(d.line);
  if (_logs[d.channel].length > 400) _logs[d.channel].shift();
  if (d.channel === _curLog) _appendLine(d.channel, d.line);
});
socket.on('clear_logs', () => {
  _logs = { log_main:[], log_coords:[], log_collect:[] };
  document.getElementById('log-panel').innerHTML = '';
});
socket.on('image', d => {
  const now = Date.now();
  const isDetect = d.channel === 'img_detect';
  const imgEl = document.getElementById(isDetect ? 'img-detect' : 'img-depth');
  const phEl  = document.getElementById(isDetect ? 'ph-detect'  : 'ph-depth');
  const fpsEl = document.getElementById(isDetect ? 'fps-detect' : 'fps-depth');
  imgEl.src = 'data:image/jpeg;base64,' + d.data;
  imgEl.style.display = 'block';
  phEl.style.display  = 'none';
  const dt = now - (_imgTs[d.channel] || now);
  if (dt > 0) fpsEl.textContent = (1000/dt).toFixed(1) + ' fps';
  _imgTs[d.channel] = now;
});
 
function setDot(name, on) {
  ['dot-'+name,'s-'+name].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('on', on);
    el.classList.toggle('err', !on);
    el.style.background = on ? 'var(--green)' : 'var(--red)';
  });
  if (name === 'ros') document.getElementById('s-pub').style.background = on ? 'var(--green)' : 'var(--dim)';
}
function sendCmd(cmd, extra) { socket.emit('command', Object.assign({cmd}, extra||{})); }
 
// ══════════════════ SCREEN NAV ══════════════════
function goScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if (id === 'log-screen')      { _renderLog(); }
  if (id === 'settings-screen') { _loadCurrentSet(); }
}
function goHome() {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById('home').classList.add('active');
}
 
// ══════════════════ LOG ══════════════════
function switchLog(ch) {
  _curLog = ch;
  ['main','coords','collect'].forEach(k => {
    document.getElementById('lt-'+k).classList.toggle('active', ch === 'log_'+k);
  });
  _renderLog();
}
function _renderLog() {
  const panel = document.getElementById('log-panel');
  panel.innerHTML = '';
  (_logs[_curLog]||[]).forEach(l => _appendLine(_curLog, l, false));
  panel.scrollTop = panel.scrollHeight;
}
function _appendLine(ch, line, scroll=true) {
  if (ch !== _curLog) return;
  const panel = document.getElementById('log-panel');
  const d = document.createElement('div');
  d.className = 'log-line' + (ch==='log_coords'?' coord': ch==='log_collect'?' collect':'');
  d.textContent = line;
  panel.appendChild(d);
  if (scroll) panel.scrollTop = panel.scrollHeight;
}
 
// ══════════════════ SETTINGS ══════════════════
let _setCtx = 'setup';
let _setupData={}, _setupLabels={};
let _depthData={}, _depthLabels={}, _depthMethod='bm';
 
function switchSet(ctx) {
  _setCtx = ctx;
  document.getElementById('st-setup').classList.toggle('active', ctx==='setup');
  document.getElementById('st-depth').classList.toggle('active', ctx==='depth');
  document.getElementById('set-title').textContent = ctx==='setup' ? 'Robot Setup' : 'Depth Config';
  document.getElementById('method-bar').style.display = ctx==='depth' ? 'flex' : 'none';
  closeEditor();
  _loadCurrentSet();
}
function _loadCurrentSet() {
  if (_setCtx === 'setup') {
    fetch('/api/config/setup').then(r=>r.json()).then(d => {
      _setupData=d.data||{}; _setupLabels=d.labels||{};
      _renderParams(_setupData, _setupLabels, 'setup');
    });
  } else {
    fetch('/api/config/depth').then(r=>r.json()).then(d => {
      _depthData=d.data||{}; _depthLabels=d.labels||{}; _depthMethod=d.method||'bm';
      document.getElementById('btn-bm').classList.toggle('active', _depthMethod==='bm');
      document.getElementById('btn-sgbm').classList.toggle('active', _depthMethod==='sgbm');
      _renderParams(_depthData, _depthLabels, 'depth');
    });
  }
}
function setDepthMethod(m) {
  _depthMethod = m;
  fetch('/api/config/depth', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({stereo_method:m})}).then(()=>_loadCurrentSet());
}
 
function _renderParams(data, labels, ctx) {
  const list = document.getElementById('param-list');
  list.innerHTML = '';
  let idx = 0;
  for (const [key, label] of Object.entries(labels)) {
    idx++;
    const val  = data[key];
    const type = typeof val==='boolean' ? 'BOOL' : Array.isArray(val) ? 'LIST' : 'NUM';
    const tcol = type==='BOOL'?'#ffd166': type==='LIST'?'#39ff7e':'#00e5ff';
    const vcol = type==='BOOL'?'var(--yellow)': type==='LIST'?'var(--green)':'var(--accent)';
 
    const row = document.createElement('div');
    row.className = 'param-row'; row.dataset.key = key;
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
          <span class="val" style="color:${vcol}" id="pv-${key}">${_fv(val)}</span>
        </div>
      </div>`;
    row.addEventListener('click', () => _openEditor(key));
    list.appendChild(row);
  }
}
 
function _fv(v) {
  if (v===null||v===undefined) return '—';
  if (typeof v==='boolean') return v?'true':'false';
  if (Array.isArray(v)) {
    const s='['+v.map(x=>typeof x==='number'&&!Number.isInteger(x)?x.toFixed(3):String(x)).join(', ')+']';
    return s.length>34 ? s.slice(0,31)+'…]' : s;
  }
  return typeof v==='number'?(Number.isInteger(v)?String(v):v.toFixed(4)):String(v);
}
function _fvFull(v) {
  if (Array.isArray(v)) return '['+v.map(x=>typeof x==='number'&&!Number.isInteger(x)?x.toFixed(3):String(x)).join(', ')+']';
  return _fv(v);
}
 
function _openEditor(key) {
  const data   = _setCtx==='setup' ? _setupData   : _depthData;
  const labels = _setCtx==='setup' ? _setupLabels : _depthLabels;
  const val    = data[key];
  const label  = labels[key]||key;
 
  // highlight row
  document.querySelectorAll('.param-row').forEach(r => {
    r.classList.toggle('sel', r.dataset.key===key);
    r.querySelector('.accent-bar').style.background = r.dataset.key===key ? 'var(--accent)' : 'transparent';
  });
 
  document.getElementById('ed-title').textContent  = label;
  document.getElementById('ed-key').textContent    = key;
  document.getElementById('ed-curval').textContent = _fvFull(val);
 
  const area = document.getElementById('ed-input-area');
  area.innerHTML = '';
 
  if (typeof val === 'boolean') {
    let cur = val;
    const br = document.createElement('div'); br.className='bool-row';
    const bon = document.createElement('button'); bon.className='bool-btn'+(cur?' sel-on':''); bon.textContent='● ON';
    const bof = document.createElement('button'); bof.className='bool-btn'+(cur?'':' sel-off'); bof.textContent='● OFF';
    bon.onclick = ()=>{ cur=true;  bon.className='bool-btn sel-on'; bof.className='bool-btn'; };
    bof.onclick = ()=>{ cur=false; bof.className='bool-btn sel-off'; bon.className='bool-btn'; };
    br.append(bon,bof); area.appendChild(br);
    const sv=document.createElement('button'); sv.className='save-btn'; sv.textContent='💾  SAVE';
    sv.onclick=()=>_saveParam(key,cur); area.appendChild(sv);
  } else {
    const isList = Array.isArray(val);
    const lbl=document.createElement('div'); lbl.className='inp-label';
    lbl.textContent=isList?'Edit list (comma-separated):':'New value:';
    area.appendChild(lbl);
    const wrap=document.createElement('div'); wrap.className='inp-wrap';
    const inp=document.createElement('input'); inp.type='text';
    inp.value=isList?val.join(', '):String(val);
    wrap.appendChild(inp); area.appendChild(wrap);
    const sv=document.createElement('button'); sv.className='save-btn'; sv.textContent='💾  SAVE';
    sv.onclick=()=>{
      let nv;
      if(isList){ try{nv=inp.value.split(',').map(p=>{const t=p.trim();return t.includes('.')?parseFloat(t):parseInt(t,10)});}catch(e){toast('Invalid list');return;} }
      else { nv=Number.isInteger(val)?parseInt(inp.value,10):parseFloat(inp.value); if(isNaN(nv)){toast('Invalid number');return;} }
      _saveParam(key,nv);
    };
    area.appendChild(sv);
    inp.focus();
  }
 
  document.getElementById('param-list').style.display='none';
  const ed=document.getElementById('param-editor');
  ed.style.display='flex'; ed.classList.add('active');
}
 
function closeEditor() {
  document.getElementById('param-editor').style.display='none';
  document.getElementById('param-editor').classList.remove('active');
  document.getElementById('param-list').style.display='';
  document.querySelectorAll('.param-row').forEach(r=>{
    r.classList.remove('sel');
    r.querySelector('.accent-bar').style.background='transparent';
  });
}
 
function _saveParam(key, val) {
  const body={[key]:val};
  if(_setCtx==='depth') body.stereo_method=_depthMethod;
  fetch(`/api/config/${_setCtx}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  .then(r=>r.json()).then(d=>{
    if(d.ok){
      if(_setCtx==='setup') _setupData[key]=val; else _depthData[key]=val;
      const el=document.getElementById('pv-'+key);
      if(el) el.textContent=_fv(val);
      document.getElementById('ed-curval').textContent=_fvFull(val);
      toast('✓  Saved');
    } else toast('Error: '+d.error);
  });
}
 
// ══════════════════ QR ══════════════════
let _qrLoaded = false;
function _loadQR() {
  fetch('/api/qr').then(r=>r.json()).then(d=>{
    if(!d.qr) return;
    document.getElementById('qr-img').src='data:image/png;base64,'+d.qr;
    const badge=document.getElementById('qr-badge');
    badge.textContent = d.is_public ? '🌐  PUBLIC — mọi nơi' : '📶  LOCAL — cùng WiFi';
    badge.style.color = d.is_public ? 'var(--green)' : 'var(--yellow)';
    document.getElementById('qr-url-text').textContent = d.url;
    _qrLoaded=true;
  }).catch(()=>{});
}
function showQR() { _qrLoaded=false; _loadQR(); document.getElementById('qr-overlay').classList.add('show'); }
function closeQR() { document.getElementById('qr-overlay').classList.remove('show'); }
 
// ══════════════════ TOAST ══════════════════
let _tt;
function toast(msg,dur=1800){
  const t=document.getElementById('toast');
  t.textContent=msg; t.style.display='block';
  clearTimeout(_tt); _tt=setTimeout(()=>t.style.display='none',dur);
}
 
// ══════════════════ INIT ══════════════════
document.addEventListener('DOMContentLoaded',()=>{
  fetch('/api/logs').then(r=>r.json()).then(d=>{
    for(const ch of Object.keys(d)) _logs[ch]=d[ch]||[];
  });
  _loadQR();
});
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

    # Register bus→WebSocket bridge (must be after _bus_to_ws is defined)
    bus.subscribe(_bus_to_ws)
    print('[SERVER] Log bus connected to WebSocket')

    sio.run(app, host='0.0.0.0', port=port,
            debug=False, use_reloader=False,
            allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()