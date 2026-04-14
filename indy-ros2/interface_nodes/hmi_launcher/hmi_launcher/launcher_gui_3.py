#!/usr/bin/env python3
"""
Indy ROS2 HMI — Tkinter version
Entry point: main()
"""

import sys
import os
import signal as _signal
import subprocess
import threading
import queue
import time
import yaml
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
from hmi_launcher.hmi_qr import show_qr_window

# ── Optional ROS2 ─────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image as RosImage
    from start_msgs.msg import StartMsg
    import numpy as np
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    np = None

try:
    from hmi_launcher.log_bus import bus as _log_bus
except ImportError:
    _log_bus = None

# ─── Config paths ─────────────────────────────────────────────────────────────
def _find_config() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        candidate = parent / 'src' / 'indy-ros2' / 'msg' / 'config_manager' / 'config'
        if candidate.exists():
            return candidate
    return Path("/home/nhut/indy-ros2/config")

_CFG = _find_config()
SETUP_PATH        = _CFG / 'setup.yaml'
DEPTH_CONFIG_PATH = _CFG / 'stereo_config.yaml'

IMAGE_TOPIC_DETECT = "/stereo/left/image_yolo"
IMAGE_TOPIC_DEPTH  = "/stereo/depth/image_raw"
START_TOPIC        = "/start_msg"

AUTO_CMD = (
    "source /opt/ros/humble/setup.bash &&"
    "ros2 launch indy_moveit indy_moveit_gazebo.launch.py indy_type:=indy7"
)

DETECTOR_KW = "detector"
COORD_KW    = "Updated target:"
COLLECT_KW  = "collect_logger_node"

LOG_MAX_LINES = 400
PROC_Q_MAX    = 300
FULLSCREEN = False

# ── Config labels ─────────────────────────────────────────────────────────────
SETUP_LABELS = {
    "HomePose":           "Home Pose (joint list)",
    "DorpPose":           "Drop Pose (joint list)",
    "OffSetDistance":     "Offset Distance (m)",
    "YOffSetDistance":    "Y Offset Distance (m)",
    "OffSetAngle":        "Offset Angle (rad)",
    "FxOffset":           "Fx Offset (m)",
    "ObjectOffset":       "Object Offset (m)",
    "Multi_collect_mode": "Multi Collect Mode (bool)",
}
STEREO_BM_LABELS = {
    "numDisparities":   "Num Disparities",
    "blockSize":        "Block Size (odd≥1)",
    "preFilterType":    "Pre-Filter Type",
    "preFilterSize":    "Pre-Filter Size",
    "preFilterCap":     "Pre-Filter Cap",
    "textureThreshold": "Texture Threshold",
    "uniquenessRatio":  "Uniqueness Ratio (%)",
    "speckleWindowSize":"Speckle Window",
    "speckleRange":     "Speckle Range",
    "disp12MaxDiff":    "Disp12 Max Diff",
}
STEREO_SGBM_LABELS = {
    "minDisparity":     "Min Disparity",
    "numDisparities":   "Num Disparities",
    "blockSize":        "Block Size (odd≥1)",
    "P1":               "P1 (SGBM penalty)",
    "P2":               "P2 (SGBM penalty)",
    "disp12MaxDiff":    "Disp12 Max Diff",
    "uniquenessRatio":  "Uniqueness Ratio (%)",
    "speckleWindowSize":"Speckle Window",
    "speckleRange":     "Speckle Range",
    "preFilterCap":     "Pre-Filter Cap",
    "mode":             "Algorithm Mode",
}

# ══════════════════════════════════════════════════════════════════════════════
# ── CENTRALISED STYLE — chỉnh tại đây để thay đổi toàn bộ giao diện
# ══════════════════════════════════════════════════════════════════════════════

# ── Colors ────────────────────────────────────────────────────────────────────
BG_COLOR     = "#0d0f14"
PANEL_COLOR  = "#131720"
ACCENT_COLOR = "#00e5ff"
GREEN_COLOR  = "#39ff7e"
RED_COLOR    = "#ff3b5c"
YELLOW_COLOR = "#ffd166"
TEXT_COLOR   = "#c8d6e5"
DIM_COLOR    = "#4a5568"

# ── Font sizes ────────────────────────────────────────────────────────────────
FS_XS   = 10   # timestamp, minor dim labels
FS_SM   = 12   # secondary labels, captions
FS_MD   = 14   # body text, log entries, buttons
FS_LG   = 16   # section headers, row labels
FS_XL   = 20   # main title, big controls
FS_2XL  = 24   # extra-large display elements

# ── Font family ───────────────────────────────────────────────────────────────
FONT     = "Courier New"

# ── Convenience font tuples ───────────────────────────────────────────────────
F_XS      = (FONT, FS_XS)
F_XS_B    = (FONT, FS_XS, "bold")
F_SM      = (FONT, FS_SM)
F_SM_B    = (FONT, FS_SM, "bold")
F_MD      = (FONT, FS_MD)
F_MD_B    = (FONT, FS_MD, "bold")
F_LG      = (FONT, FS_LG)
F_LG_B    = (FONT, FS_LG, "bold")
F_XL      = (FONT, FS_XL)
F_XL_B    = (FONT, FS_XL, "bold")
F_2XL_B   = (FONT, FS_2XL, "bold")

# ── Setup-dialog specific sizes ───────────────────────────────────────────────
SETUP_ROW_FONT      = F_LG_B   # label in parameter row
SETUP_ROW_KEY_FONT  = F_MD     # key name in parameter row
SETUP_ROW_VAL_FONT  = F_LG_B   # value in parameter row
SETUP_ROW_IDX_FONT  = F_MD     # index number
SETUP_ROW_BADGE_FONT= F_SM_B   # NUM / BOOL / LIST badge
SETUP_ROW_PAD_Y     = 12       # vertical padding inside each row
SETUP_HDR_FONT      = F_LG_B   # "SELECT PARAMETER" header
SETUP_EDITOR_TITLE  = F_LG_B   # param label in editor
SETUP_EDITOR_KEY    = F_MD     # key name in editor
SETUP_EDITOR_CURVAL = F_LG_B   # current-value display
SETUP_EDITOR_INPUT  = F_LG_B   # input entry widget
SETUP_EDITOR_BTN    = F_MD_B   # save button
SETUP_EDITOR_LBL    = F_MD     # "New value:" labels
SETUP_KBD_FONT      = F_LG_B   # keyboard keys
SETUP_KBD_SP_FONT   = F_MD_B   # SHIFT/CAPS/SPACE keys

# ── Main-window button style ──────────────────────────────────────────────────
BUTTON_STYLE = {
    "font":   F_LG_B,
    "relief": tk.FLAT,
    "bd":     0,
    "padx":   10,
    "pady":   8,
    "cursor": "hand2",
}

def style_button(btn, bg, fg):
    btn.configure(bg=bg, fg=fg, **BUTTON_STYLE)
    btn.bind("<Enter>", lambda e: btn.configure(bg="#1e2535"))
    btn.bind("<Leave>", lambda e: btn.configure(bg=bg))

# ══════════════════════════════════════════════════════════════════════════════

# ── Log Widget ────────────────────────────────────────────────────────────────
class LogWidget(scrolledtext.ScrolledText):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, wrap=tk.WORD, font=F_MD,
                         bg=BG_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                         relief=tk.FLAT, bd=0, **kwargs)
        self.config(state=tk.DISABLED)
        self._queue = queue.Queue()
        self._after_id = None

    def append_safe(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self._queue.put(f"[{ts}] {text}\n")
        if self._after_id is None:
            self._after_id = self.after(50, self._update)

    def _update(self):
        self._after_id = None
        self.config(state=tk.NORMAL)
        count = 0
        while not self._queue.empty() and count < 20:
            line = self._queue.get()
            self.insert(tk.END, line)
            count += 1
        if int(self.index('end-1c').split('.')[0]) > LOG_MAX_LINES:
            self.delete(1.0, 2.0)
        self.see(tk.END)
        self.config(state=tk.DISABLED)
        if not self._queue.empty():
            self._after_id = self.after(50, self._update)

    def clear(self):
        self.config(state=tk.NORMAL)
        self.delete(1.0, tk.END)
        self.config(state=tk.DISABLED)

# ── Log Panel ─────────────────────────────────────────────────────────────────
class LogPanel(tk.Frame):
    def __init__(self, parent, title, color):
        super().__init__(parent, bg=PANEL_COLOR, bd=0)
        self._count = 0
        header = tk.Frame(self, bg="#0a0c11", height=34)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="●", fg=color, bg="#0a0c11", font=F_SM).pack(side=tk.LEFT, padx=(8, 4))
        tk.Label(header, text=title.upper(), fg=color, bg="#0a0c11", font=F_SM_B).pack(side=tk.LEFT)
        self.count_lbl = tk.Label(header, text="0", fg=DIM_COLOR, bg="#1e2535",
                                  font=F_SM, padx=5, pady=1)
        self.count_lbl.pack(side=tk.RIGHT, padx=8)
        self.log = LogWidget(self, height=12)
        self.log.pack(fill=tk.BOTH, expand=True)
        self._title_str = title.upper()

    def append(self, line):
        self._count += 1
        self.count_lbl.config(text=str(self._count))
        self.log.append_safe(line)
        if _log_bus is not None:
            ch = {'COORDINATES':'log_coords','COLLECT':'log_collect'}.get(self._title_str,'log_main')
            _log_bus.push(ch, line)

    def clear(self):
        self._count = 0
        self.count_lbl.config(text="0")
        self.log.clear()

# ── Camera Widget ─────────────────────────────────────────────────────────────
class CameraWidget(tk.Frame):
    def __init__(self, parent, title, topic):
        super().__init__(parent, bg=PANEL_COLOR)
        self.topic = topic
        self._last_t = None
        header = tk.Frame(self, bg="#0a0c11", height=28)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="●", fg=ACCENT_COLOR, bg="#0a0c11", font=F_XS).pack(side=tk.LEFT, padx=(8, 4))
        tk.Label(header, text=title, fg=ACCENT_COLOR, bg="#0a0c11", font=F_SM_B).pack(side=tk.LEFT)
        self.fps_lbl = tk.Label(header, text="no signal", fg=DIM_COLOR, bg="#0a0c11", font=F_SM)
        self.fps_lbl.pack(side=tk.RIGHT, padx=8)
        self.canvas = tk.Canvas(self, bg="#050608", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._photo = None

    def update_image(self, qimg):
        try:
            if isinstance(qimg, np.ndarray):
                img = Image.fromarray(qimg)
            else:
                return
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w < 10 or h < 10:
                w, h = 320, 240
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=self._photo)
            now = time.time()
            if self._last_t:
                fps = 1 / (now - self._last_t)
                self.fps_lbl.config(text=f"{fps:.1f} fps")
            self._last_t = now
        except Exception as e:
            print(f"Camera error: {e}")

# ── ROS Worker ────────────────────────────────────────────────────────────────
class RosWorker:
    def __init__(self, node, executor):
        self.node = node
        self.executor = executor
        self.running = True
        self.last_decode = {}
        self.min_interval = 0.1

    def spin(self):
        while self.running:
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except:
                break

    def stop(self):
        self.running = False
        try:
            self.executor.shutdown()
        except:
            pass

    def decode_image(self, topic, msg):
        now = time.time()
        if now - self.last_decode.get(topic, 0) < self.min_interval:
            return None
        self.last_decode[topic] = now
        try:
            enc = msg.encoding.lower()
            raw = np.frombuffer(msg.data, dtype=np.uint8).copy()
            if enc == "mono8":
                arr = raw.reshape((msg.height, msg.width))
                arr = np.stack([arr, arr, arr], axis=2)
            elif enc == "rgb8":
                arr = raw.reshape((msg.height, msg.width, 3))
            elif enc == "bgr8":
                arr = raw.reshape((msg.height, msg.width, 3))[:, :, ::-1].copy()
            else:
                arr = raw.reshape((msg.height, msg.width, -1))[:, :, :3].copy()
            return arr
        except Exception as e:
            print(f"Decode error {topic}: {e}")
            return None

# ── helper ────────────────────────────────────────────────────────────────────
def _set_bg_recursive(widget, bg):
    """Recursively set background on a widget tree (skips Entries/Buttons)."""
    try:
        wclass = widget.winfo_class()
        if wclass not in ("Entry", "Button", "Radiobutton", "TScrollbar"):
            widget.config(bg=bg)
    except:
        pass
    for child in widget.winfo_children():
        _set_bg_recursive(child, bg)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Setup Dialog
# ═══════════════════════════════════════════════════════════════════════════════
class SetupDialog(tk.Toplevel):
    """
    Screen 1 — Parameter list (full screen, vertical scroll)
    Screen 2 — Editor: form top | keyboard bottom  (replaces list on click)
    ← BACK returns to list.
    """

    _C = {
        "bg":        "#0b0d12",
        "sidebar":   "#0e1118",
        "card":      "#161b26",
        "card_sel":  "#1a2235",
        "border":    "#1e2840",
        "border_hi": "#00e5ff",
        "accent":    "#00e5ff",
        "green":     "#39ff7e",
        "red":       "#ff3b5c",
        "yellow":    "#ffd166",
        "dim":       "#3d4f68",
        "text":      "#c8d6e5",
        "subtext":   "#7a8fa8",
        "kbd_bg":    "#0e1118",
        "kbd_key":   "#1c2333",
        "kbd_key_h": "#00e5ff",
        "editor_bg": "#10131a",
    }

    KBD_H = 320   # height of the keyboard strip at the bottom of edit screen

    def __init__(self, parent, path, labels, yaml_section=None):
        super().__init__(parent)
        self.title(f"SETUP — {path.name}")
        self.attributes('-fullscreen', FULLSCREEN)
        self.configure(bg=self._C["bg"])
        self.path = path
        self.labels = labels
        self.section = yaml_section
        self.data = {}
        self.current_key = None
        self._edit_mode = False
        self._param_cards = {}
        self._load()
        self._build()

    # ── data I/O ─────────────────────────────────────────────────────────────
    def _load(self):
        try:
            with open(self.path, 'r') as f:
                raw = yaml.safe_load(f) or {}
            src = raw.get(self.section, raw) if self.section else raw
            self.data = dict(src)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _save(self):
        try:
            raw = yaml.safe_load(self.path.read_text()) or {}
            sec = raw.setdefault(self.section, {}) if self.section else raw
            sec.update(self.data)
            with open(self.path, 'w') as f:
                yaml.dump(raw, f, allow_unicode=True, sort_keys=False)
            messagebox.showinfo("Saved", "Configuration saved.")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _fv(self, v):
        if isinstance(v, list):
            inner = ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v)
            full  = f"[{inner}]"
            return full if len(full) <= 32 else full[:29] + "…]"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def _fv_full(self, v):
        if isinstance(v, list):
            return "[" + ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v) + "]"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    # ── top-level build ───────────────────────────────────────────────────────
    def _build(self):
        C = self._C

        # shared header (always visible)
        self._header = tk.Frame(self, bg="#080a0f", height=56)
        self._header.pack(fill=tk.X, side=tk.TOP)
        self._header.pack_propagate(False)
        self._build_header(self._header)

        tk.Frame(self, bg=C["border_hi"], height=2).pack(fill=tk.X)

        # ── list screen ───────────────────────────────────────────────────────
        self._top = tk.Frame(self, bg=C["sidebar"])
        self._top.pack(fill=tk.BOTH, expand=True)
        self._build_param_list()

        # ── editor screen (hidden initially) ──────────────────────────────────
        self._bottom = tk.Frame(self, bg=C["editor_bg"])
        # not packed yet — shown on demand

    def _build_header(self, header):
        C = self._C
        tk.Label(header, text="⚙", fg=C["accent"], bg="#080a0f",
                 font=F_XL_B).pack(side=tk.LEFT, padx=(20, 6), pady=8)
        tk.Label(header, text=self.path.name.upper(), fg=C["accent"], bg="#080a0f",
                 font=F_LG_B).pack(side=tk.LEFT, pady=8)
        tk.Label(header, text=f"  •  {len(self.labels)} parameters",
                 fg=C["dim"], bg="#080a0f", font=F_SM).pack(side=tk.LEFT, pady=8)

        btn_close = tk.Button(header, text="✕  CLOSE", command=self.destroy,
                              bg=C["red"], fg="white", font=F_MD_B,
                              relief=tk.FLAT, padx=18, pady=10, cursor="hand2", bd=0)
        btn_close.pack(side=tk.RIGHT, padx=20, pady=10)
        btn_close.bind("<Enter>", lambda e: btn_close.config(bg="#cc2240"))
        btn_close.bind("<Leave>", lambda e: btn_close.config(bg=C["red"]))

    # ── parameter list ────────────────────────────────────────────────────────
    def _build_param_list(self):
        C = self._C
        container = self._top

        # sub-header
        sub_hdr = tk.Frame(container, bg=C["sidebar"], height=36)
        sub_hdr.pack(fill=tk.X, side=tk.TOP)
        sub_hdr.pack_propagate(False)
        tk.Label(sub_hdr, text="  SELECT PARAMETER", fg=C["dim"], bg=C["sidebar"],
                 font=SETUP_HDR_FONT).pack(side=tk.LEFT, padx=12, pady=6)
        tk.Label(sub_hdr, text=str(len(self.labels)), fg=C["accent"], bg=C["sidebar"],
                 font=SETUP_HDR_FONT).pack(side=tk.RIGHT, padx=12)

        tk.Frame(container, bg=C["border"], height=1).pack(fill=tk.X)

        # scrollable canvas
        v_scroll = tk.Scrollbar(container, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        v_canvas = tk.Canvas(container, bg=C["sidebar"],
                             highlightthickness=0, bd=0,
                             yscrollcommand=v_scroll.set)
        v_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.configure(command=v_canvas.yview)

        rows_frame = tk.Frame(v_canvas, bg=C["sidebar"])
        _win = v_canvas.create_window((0, 0), window=rows_frame, anchor="nw")

        def _on_frame_configure(e):
            v_canvas.configure(scrollregion=v_canvas.bbox("all"))

        def _on_canvas_configure(e):
            # ← KEY FIX: force rows_frame to always match canvas width
            v_canvas.itemconfig(_win, width=e.width)

        rows_frame.bind("<Configure>", _on_frame_configure)
        v_canvas.bind("<Configure>",   _on_canvas_configure)

        def _vscroll(e):
            v_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        v_canvas.bind("<MouseWheel>",  _vscroll)
        rows_frame.bind("<MouseWheel>", _vscroll)

        self._param_cards = {}
        for idx, (key, label) in enumerate(self.labels.items()):
            self._make_row(rows_frame, key, label, idx)

    def _make_row(self, parent, key, label, idx):
        C   = self._C
        val = self.data.get(key)

        is_bool = isinstance(val, bool)
        is_list = isinstance(val, list)

        if is_bool:
            badge_text, badge_bg, type_color = "BOOL", "#ffd166", C["yellow"]
        elif is_list:
            badge_text, badge_bg, type_color = "LIST", "#39ff7e", C["green"]
        else:
            badge_text, badge_bg, type_color = "NUM",  "#00e5ff", C["accent"]

        # ── outer row — fills full width ──────────────────────────────────────
        row = tk.Frame(parent, bg=C["card"], cursor="hand2", bd=0)
        row.pack(fill=tk.X)          # no side=, no padx/pady so it truly fills

        # left accent bar
        accent_bar = tk.Frame(row, bg=C["card"], width=4)
        accent_bar.pack(side=tk.LEFT, fill=tk.Y)
        accent_bar.pack_propagate(False)

        # inner content — expand=True so it stretches horizontally
        inner = tk.Frame(row, bg=C["card"], padx=14, pady=SETUP_ROW_PAD_Y)
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── top line: index  |  label  |  badge ──────────────────────────────
        top_line = tk.Frame(inner, bg=C["card"])
        top_line.pack(fill=tk.X)

        tk.Label(top_line, text=f"{idx+1:02d}", fg=C["dim"], bg=C["card"],
                 font=SETUP_ROW_IDX_FONT).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(top_line, text=label, fg=C["text"], bg=C["card"],
                 font=SETUP_ROW_FONT, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(top_line, text=badge_text, fg="#0d0f14", bg=badge_bg,
                 font=SETUP_ROW_BADGE_FONT, padx=6, pady=2).pack(side=tk.RIGHT)

        # ── bottom line: key (left)  |  value (right) ────────────────────────
        bot_line = tk.Frame(inner, bg=C["card"])
        bot_line.pack(fill=tk.X, pady=(4, 0))

        tk.Label(bot_line, text=key, fg=C["dim"], bg=C["card"],
                 font=SETUP_ROW_KEY_FONT).pack(side=tk.LEFT)

        val_lbl = tk.Label(bot_line, text=self._fv(val), fg=type_color, bg=C["card"],
                           font=SETUP_ROW_VAL_FONT)
        val_lbl.pack(side=tk.RIGHT)

        # thin separator
        tk.Frame(parent, bg=C["border"], height=1).pack(fill=tk.X)

        # ── hover / click bindings ────────────────────────────────────────────
        def _enter(e, r=row, a=accent_bar):
            r.config(bg=C["card_sel"])
            _set_bg_recursive(r, C["card_sel"])
            a.config(bg=C["accent"])

        def _leave(e, r=row, a=accent_bar, k=key):
            sel = (self.current_key == k)
            bg  = C["card_sel"] if sel else C["card"]
            ac  = C["accent"]   if sel else C["card"]
            r.config(bg=bg)
            _set_bg_recursive(r, bg)
            a.config(bg=ac)

        def _click(e, k=key):
            self._select_card(k)

        for w in [row, inner, top_line, bot_line, accent_bar] + \
                 list(top_line.winfo_children()) + \
                 list(bot_line.winfo_children()):
            try:
                w.bind("<Enter>",    _enter)
                w.bind("<Leave>",    _leave)
                w.bind("<Button-1>", _click)
            except Exception:
                pass

        self._param_cards[key] = {
            "card": row, "stripe": accent_bar,
            "val_lbl": val_lbl, "type_color": type_color,
        }

    # ── card state ────────────────────────────────────────────────────────────
    def _select_card(self, key):
        C = self._C
        if self.current_key and self.current_key in self._param_cards:
            prev = self._param_cards[self.current_key]
            prev["card"].config(bg=C["card"])
            _set_bg_recursive(prev["card"], C["card"])
            prev["stripe"].config(bg=C["card"])
        self.current_key = key
        cur = self._param_cards[key]
        cur["card"].config(bg=C["card_sel"])
        _set_bg_recursive(cur["card"], C["card_sel"])
        cur["stripe"].config(bg=C["accent"])
        self._enter_edit_mode(key)

    def _update_card_value(self, key, new_str):
        if key in self._param_cards:
            self._param_cards[key]["val_lbl"].config(text=new_str)

    # ── screen transitions ────────────────────────────────────────────────────
    def _enter_edit_mode(self, key):
        if self._edit_mode:
            self._show_editor(key)
            return
        self._edit_mode = True
        self._top.pack_forget()
        self._bottom.pack(fill=tk.BOTH, expand=True)
        self._show_editor(key)

    def _exit_edit_mode(self):
        self._edit_mode = False
        self._bottom.pack_forget()
        self._top.pack(fill=tk.BOTH, expand=True)
        # deselect
        if self.current_key and self.current_key in self._param_cards:
            prev = self._param_cards[self.current_key]
            prev["card"].config(bg=self._C["card"])
            _set_bg_recursive(prev["card"], self._C["card"])
            prev["stripe"].config(bg=self._C["card"])
        self.current_key = None

    # ── editor screen ─────────────────────────────────────────────────────────
    def _show_editor(self, key):
        C = self._C
        for w in self._bottom.winfo_children():
            w.destroy()

        val       = self.data.get(key)
        label     = self.labels.get(key, key)
        has_kbd   = not isinstance(val, bool)

        # ── BACK button ───────────────────────────────────────────────────────
        back_row = tk.Frame(self._bottom, bg=C["editor_bg"])
        back_row.pack(fill=tk.X, padx=20, pady=(12, 0))

        back_btn = tk.Button(back_row, text="←  BACK",
                             command=self._exit_edit_mode,
                             bg=C["border"], fg=C["text"], font=F_MD_B,
                             relief=tk.FLAT, padx=14, pady=8, cursor="hand2", bd=0)
        back_btn.pack(side=tk.LEFT)
        back_btn.bind("<Enter>", lambda e: back_btn.config(bg=C["dim"]))
        back_btn.bind("<Leave>", lambda e: back_btn.config(bg=C["border"]))

        # ── keyboard strip (bottom, fixed height) ─────────────────────────────
        entry_ref = [None]
        if has_kbd:
            tk.Frame(self._bottom, bg=C["border"], height=1).pack(
                side=tk.BOTTOM, fill=tk.X)
            kb_frame = tk.Frame(self._bottom, bg=C["kbd_bg"], height=self.KBD_H)
            kb_frame.pack(side=tk.BOTTOM, fill=tk.X)
            kb_frame.pack_propagate(False)
            # keyboard is wired after entry is created — store frame reference
            self._kb_frame = kb_frame

        # ── form area (fills between back-row and keyboard) ────────────────────
        form = tk.Frame(self._bottom, bg=C["editor_bg"])
        form.pack(fill=tk.BOTH, expand=True, padx=30, pady=14)

        # param title
        name_row = tk.Frame(form, bg=C["editor_bg"])
        name_row.pack(fill=tk.X, pady=(0, 10))
        tk.Label(name_row, text=label, fg=C["accent"], bg=C["editor_bg"],
                 font=SETUP_EDITOR_TITLE).pack(side=tk.LEFT)
        tk.Label(name_row, text=f"  ·  {key}", fg=C["dim"], bg=C["editor_bg"],
                 font=SETUP_EDITOR_KEY).pack(side=tk.LEFT, pady=(4, 0))

        # current value pill
        pill = tk.Frame(form, bg=C["card"])
        pill.pack(fill=tk.X, pady=(0, 14))
        tk.Frame(pill, bg=C["accent"], width=4).pack(side=tk.LEFT, fill=tk.Y)
        pill_inner = tk.Frame(pill, bg=C["card"], padx=16, pady=12)
        pill_inner.pack(fill=tk.X, expand=True)
        tk.Label(pill_inner, text="CURRENT VALUE", fg=C["dim"], bg=C["card"],
                 font=F_XS_B).pack(anchor="w")
        tk.Label(pill_inner, text=self._fv_full(val), fg=C["yellow"], bg=C["card"],
                 font=SETUP_EDITOR_CURVAL, wraplength=900,
                 justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        tk.Frame(form, bg=C["border"], height=1).pack(fill=tk.X, pady=(0, 14))

        # input widgets
        if isinstance(val, bool):
            self._build_bool_editor(form, key, val, C)
        elif isinstance(val, list):
            entry_ref[0] = self._build_list_editor(form, key, val, C)
        else:
            entry_ref[0] = self._build_scalar_editor(form, key, val, C)

        # wire keyboard now that entry exists
        if has_kbd and entry_ref[0] is not None:
            self._build_embedded_keyboard(self._kb_frame, entry_ref[0], C)

    # ── bool editor ───────────────────────────────────────────────────────────
    def _build_bool_editor(self, parent, key, val, C):
        var = tk.BooleanVar(value=val)
        frm = tk.Frame(parent, bg=C["editor_bg"])
        frm.pack(fill=tk.X)
        tk.Label(frm, text="Toggle value:", fg=C["subtext"], bg=C["editor_bg"],
                 font=SETUP_EDITOR_LBL).pack(anchor="w", pady=(0, 12))
        btn_row = tk.Frame(frm, bg=C["editor_bg"])
        btn_row.pack(anchor="w")
        for text, bval, col in [("●  ON", True, C["green"]), ("●  OFF", False, C["red"])]:
            rb = tk.Radiobutton(btn_row, text=text, variable=var, value=bval,
                                bg=C["card"], fg=col, selectcolor=C["card_sel"],
                                activebackground=C["card"], activeforeground=col,
                                font=SETUP_EDITOR_BTN,
                                relief=tk.FLAT, padx=26, pady=16, cursor="hand2",
                                indicatoron=0, width=10)
            rb.pack(side=tk.LEFT, padx=(0, 14))

        def save_bool():
            self.data[key] = var.get()
            self._save()
            self._update_card_value(key, self._fv(self.data[key]))

        self._save_btn(parent, save_bool, C)

    # ── list editor ───────────────────────────────────────────────────────────
    def _build_list_editor(self, parent, key, val, C):
        frm = tk.Frame(parent, bg=C["editor_bg"])
        frm.pack(fill=tk.X)
        tk.Label(frm, text="Edit list (comma-separated):", fg=C["subtext"], bg=C["editor_bg"],
                 font=SETUP_EDITOR_LBL).pack(anchor="w", pady=(0, 10))
        ef = tk.Frame(frm, bg=C["accent"], padx=2, pady=2)
        ef.pack(fill=tk.X)
        entry = tk.Entry(ef, font=SETUP_EDITOR_INPUT,
                         bg=C["card"], fg=C["text"], insertbackground=C["accent"],
                         relief=tk.FLAT, bd=10)
        entry.pack(fill=tk.X)
        entry.insert(0, ", ".join(str(x) for x in val))

        def save_list():
            try:
                parts = entry.get().split(',')
                nv = [float(p.strip()) if '.' in p else int(p.strip()) for p in parts]
                self.data[key] = nv
                self._save()
                self._update_card_value(key, self._fv(self.data[key]))
            except Exception:
                messagebox.showerror("Error", "Invalid list format")

        self._save_btn(parent, save_list, C)
        return entry

    # ── scalar editor ─────────────────────────────────────────────────────────
    def _build_scalar_editor(self, parent, key, val, C):
        frm = tk.Frame(parent, bg=C["editor_bg"])
        frm.pack(fill=tk.X)
        tk.Label(frm, text="New value:", fg=C["subtext"], bg=C["editor_bg"],
                 font=SETUP_EDITOR_LBL).pack(anchor="w", pady=(0, 10))
        ef = tk.Frame(frm, bg=C["accent"], padx=2, pady=2)
        ef.pack(fill=tk.X)
        entry = tk.Entry(ef, font=SETUP_EDITOR_INPUT,
                         bg=C["card"], fg=C["accent"], insertbackground=C["accent"],
                         relief=tk.FLAT, bd=10)
        entry.pack(fill=tk.X)
        entry.insert(0, str(val))
        entry.icursor(tk.END)
        entry.focus_set()

        def save_scalar():
            try:
                if isinstance(val, int) and not isinstance(val, bool):
                    nv = int(entry.get())
                elif isinstance(val, float):
                    nv = float(entry.get())
                else:
                    nv = entry.get()
                self.data[key] = nv
                self._save()
                self._update_card_value(key, self._fv(self.data[key]))
            except Exception:
                messagebox.showerror("Error", f"Invalid type, expected {type(val).__name__}")

        self._save_btn(parent, save_scalar, C)
        return entry

    # ── save button ───────────────────────────────────────────────────────────
    def _save_btn(self, parent, cmd, C):
        frm = tk.Frame(parent, bg=C["editor_bg"])
        frm.pack(fill=tk.X, pady=(16, 0))
        btn = tk.Button(frm, text="💾  SAVE CHANGES", command=cmd,
                        bg=C["green"], fg="#0b0d12", font=SETUP_EDITOR_BTN,
                        relief=tk.FLAT, padx=26, pady=12, cursor="hand2", bd=0)
        btn.pack(side=tk.LEFT)
        btn.bind("<Enter>", lambda e: btn.config(bg="#2be065"))
        btn.bind("<Leave>", lambda e: btn.config(bg=C["green"]))

    # ── embedded keyboard ─────────────────────────────────────────────────────
    def _build_embedded_keyboard(self, parent, target_entry, C):
        rows = [
            "`1234567890-=",
            "qwertyuiop[]\\",
            "asdfghjkl;'",
            "zxcvbnm,./"
        ]
        shift_map = {
            "`":"~","1":"!","2":"@","3":"#","4":"$","5":"%","6":"^","7":"&",
            "8":"*","9":"(","0":")","-":"_","=":"+","[":"{","]":"}","\\":"|",
            ";":":","'":'"',",":"<",".":">","/":"?"
        }
        shift_state = [False]
        caps_state  = [False]
        all_key_btns = []

        wrapper = tk.Frame(parent, bg=C["kbd_bg"], padx=16, pady=10)
        wrapper.pack(fill=tk.BOTH, expand=True)

        def _hover_on(b):  b.config(bg=C["kbd_key_h"], fg="#0b0d12")
        def _hover_off(b): b.config(bg=C["kbd_key"],   fg=C["text"])

        def update_keys():
            active = shift_state[0] or caps_state[0]
            for ch, b in all_key_btns:
                if len(ch) == 1:
                    b.config(text=shift_map.get(ch, ch.upper()) if active else ch)

        def on_key(ch):
            active = shift_state[0] or caps_state[0]
            out = shift_map.get(ch, ch.upper()) if active else ch
            target_entry.insert(tk.INSERT, out)
            target_entry.focus_set()
            if shift_state[0] and not caps_state[0]:
                shift_state[0] = False
                update_keys()

        def toggle_shift():
            shift_state[0] = not shift_state[0]
            update_keys()

        def toggle_caps():
            caps_state[0] = not caps_state[0]
            update_keys()

        def backspace():
            try:
                pos = target_entry.index(tk.INSERT)
                if pos != "0":
                    target_entry.delete(f"{pos} -1c")
            except Exception:
                s = target_entry.get()
                if s:
                    target_entry.delete(len(s) - 1, tk.END)

        def clear_all():
            target_entry.delete(0, tk.END)

        # character rows
        for row in rows:
            rf = tk.Frame(wrapper, bg=C["kbd_bg"])
            rf.pack(pady=2)
            for ch in row:
                b = tk.Button(rf, text=ch, width=3, height=1,
                              font=SETUP_KBD_FONT,
                              bg=C["kbd_key"], fg=C["text"], relief=tk.FLAT,
                              bd=0, cursor="hand2", command=lambda c=ch: on_key(c))
                b.pack(side=tk.LEFT, padx=2)
                b.bind("<Enter>", lambda e, btn=b: _hover_on(btn))
                b.bind("<Leave>", lambda e, btn=b: _hover_off(btn))
                all_key_btns.append((ch, b))

        # special keys row
        sp = tk.Frame(wrapper, bg=C["kbd_bg"])
        sp.pack(pady=(6, 0))
        specials = [
            ("SHIFT", 6,  toggle_shift),
            ("CAPS",  6,  toggle_caps),
            ("SPACE", 16, lambda: (target_entry.insert(tk.INSERT, " "), target_entry.focus_set())),
            ("⌫",    4,  backspace),
            ("CLR",  5,  clear_all),
        ]
        for lbl, w, cmd in specials:
            b = tk.Button(sp, text=lbl, width=w, height=1,
                          font=SETUP_KBD_SP_FONT,
                          bg=C["kbd_key"], fg=C["subtext"], relief=tk.FLAT,
                          bd=0, cursor="hand2", command=cmd)
            b.pack(side=tk.LEFT, padx=3)
            b.bind("<Enter>", lambda e, btn=b: _hover_on(btn))
            b.bind("<Leave>", lambda e, btn=b: _hover_off(btn))


# ═══════════════════════════════════════════════════════════════════════════════
# ── Depth Setup Dialog
# ═══════════════════════════════════════════════════════════════════════════════
class DepthSetupDialog(SetupDialog):
    def __init__(self, parent):
        self.method = "bm"
        super().__init__(parent, DEPTH_CONFIG_PATH, STEREO_BM_LABELS, yaml_section=None)
        self.title("SETUP — stereo_config.yaml")

    def _load(self):
        try:
            with open(self.path, 'r') as f:
                raw = yaml.safe_load(f) or {}
            self.method = raw.get("stereo_method", "bm")
            self.data   = dict(raw.get(f"stereo_{self.method}", {}))
            self.labels = STEREO_SGBM_LABELS if self.method == "sgbm" else STEREO_BM_LABELS
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _build(self):
        C = self._C

        # header
        self._header = tk.Frame(self, bg="#080a0f", height=56)
        self._header.pack(fill=tk.X, side=tk.TOP)
        self._header.pack_propagate(False)

        tk.Label(self._header, text="⚙", fg=C["accent"], bg="#080a0f",
                 font=F_XL_B).pack(side=tk.LEFT, padx=(20, 6), pady=8)
        tk.Label(self._header, text="STEREO CONFIG", fg=C["accent"], bg="#080a0f",
                 font=F_LG_B).pack(side=tk.LEFT, pady=8)

        btn_close = tk.Button(self._header, text="✕  CLOSE", command=self.destroy,
                              bg=C["red"], fg="white", font=F_MD_B,
                              relief=tk.FLAT, padx=18, pady=10, cursor="hand2", bd=0)
        btn_close.pack(side=tk.RIGHT, padx=20, pady=10)
        btn_close.bind("<Enter>", lambda e: btn_close.config(bg="#cc2240"))
        btn_close.bind("<Leave>", lambda e: btn_close.config(bg=C["red"]))

        tk.Frame(self, bg=C["border_hi"], height=2).pack(fill=tk.X)

        # method bar
        method_bar = tk.Frame(self, bg="#0a0d14", height=54)
        method_bar.pack(fill=tk.X)
        method_bar.pack_propagate(False)

        tk.Label(method_bar, text="ALGORITHM:", fg=C["dim"], bg="#0a0d14",
                 font=F_MD_B).pack(side=tk.LEFT, padx=(20, 12), pady=14)

        self._bm_btn   = self._method_btn(method_bar, "BM",   "bm")
        self._sgbm_btn = self._method_btn(method_bar, "SGBM", "sgbm")
        self._refresh_method_btns()

        tk.Frame(method_bar, bg=C["dim"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=16, pady=10)
        self._method_info = tk.Label(method_bar, text=self._method_desc(),
                                     fg=C["subtext"], bg="#0a0d14", font=F_SM)
        self._method_info.pack(side=tk.LEFT, pady=14)

        tk.Frame(self, bg=C["border"], height=1).pack(fill=tk.X)

        # list screen
        self._top = tk.Frame(self, bg=C["sidebar"])
        self._top.pack(fill=tk.BOTH, expand=True)
        self._build_param_list()

        # editor screen (hidden)
        self._bottom = tk.Frame(self, bg=C["editor_bg"])

    def _method_btn(self, parent, text, method_key):
        C = self._C
        btn = tk.Button(parent, text=text, font=F_MD_B,
                        relief=tk.FLAT, padx=22, pady=8, cursor="hand2", bd=0,
                        command=lambda m=method_key: self._switch(m))
        btn.pack(side=tk.LEFT, padx=4, pady=10)
        return btn

    def _refresh_method_btns(self):
        C = self._C
        for btn, key in [(self._bm_btn, "bm"), (self._sgbm_btn, "sgbm")]:
            active = (self.method == key)
            btn.config(bg=C["accent"] if active else C["kbd_key"],
                       fg="#0d0f14"   if active else C["subtext"])

    def _method_desc(self):
        return ("Semi-Global Block Matching  •  better accuracy, slower"
                if self.method == "sgbm"
                else "Block Matching  •  faster, suited for real-time")

    def _switch(self, m):
        self.method = m
        try:
            with open(self.path, 'r') as f:
                raw = yaml.safe_load(f) or {}
            raw["stereo_method"] = m
            with open(self.path, 'w') as f:
                yaml.dump(raw, f)
        except Exception:
            pass
        self._load()
        self._refresh_method_btns()
        self._method_info.config(text=self._method_desc())
        # reset to list screen
        if self._edit_mode:
            self._exit_edit_mode()
        for w in self._top.winfo_children():
            w.destroy()
        self._param_cards = {}
        self.current_key  = None
        self._build_param_list()


# ═══════════════════════════════════════════════════════════════════════════════
# ── Virtual Keyboard (standalone – kept for compat)
# ═══════════════════════════════════════════════════════════════════════════════
class VirtualKeyboard(tk.Toplevel):
    def __init__(self, target_entry):
        super().__init__()
        self.title("Virtual Keyboard")
        self.attributes('-topmost', True)
        self.geometry("800x300")
        self.configure(bg=PANEL_COLOR)
        self.target = target_entry
        self.shift = False
        self.caps = False
        self._build()

    _shift_map = {
        "`":"~","1":"!","2":"@","3":"#","4":"$","5":"%","6":"^","7":"&",
        "8":"*","9":"(","0":")","-":"_","=":"+","[":"{","]":"}","\\":"|",
        ";":":","'":'"',",":"<",".":">","/":"?"
    }

    def _build(self):
        rows = ["`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./"]
        for row in rows:
            frm = tk.Frame(self, bg=PANEL_COLOR)
            frm.pack(pady=2)
            for ch in row:
                btn = tk.Button(frm, text=ch, width=4, height=1,
                                font=F_MD_B, bg="#222c3c", fg=TEXT_COLOR,
                                relief=tk.FLAT, bd=0,
                                command=lambda c=ch: self._type(c))
                btn.pack(side=tk.LEFT, padx=2)
        sp_frm = tk.Frame(self, bg=PANEL_COLOR)
        sp_frm.pack(pady=5)
        for lbl, w, cmd in [("SHIFT", 6, self._toggle_shift), ("CAPS", 6, self._toggle_caps),
                             ("SPACE", 20, lambda: self._insert(" ")),
                             ("⌫", 4, self._backspace), ("CLR", 5, self._clear)]:
            btn = tk.Button(sp_frm, text=lbl, width=w, height=1,
                            font=F_SM_B, bg="#222c3c", fg=TEXT_COLOR,
                            relief=tk.FLAT, command=cmd)
            btn.pack(side=tk.LEFT, padx=2)

    def _type(self, ch):
        active = self.shift or self.caps
        out = self._shift_map.get(ch, ch.upper()) if active else ch
        self._insert(out)
        if self.shift and not self.caps:
            self.shift = False

    def _insert(self, s):
        self.target.insert(tk.INSERT, s)
        self.target.focus_set()

    def _toggle_shift(self): self.shift = not self.shift
    def _toggle_caps(self):  self.caps  = not self.caps

    def _backspace(self):
        cur = self.target.index(tk.INSERT)
        if cur != "1.0":
            self.target.delete(f"{cur} -1c")

    def _clear(self): self.target.delete(0, tk.END)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main Window
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("INDY ROS2 HMI")
        self.attributes('-fullscreen', FULLSCREEN)
        self.configure(bg=BG_COLOR)
        self._proc = None
        self._proc_running = False
        self._proc_queue = queue.Queue()
        self._ros_node = None
        self._ros_worker = None
        self._start_pub = None
        self._build_ui()
        self._clock_timer()
        self._drain_timer()
        self._init_ros()
        self._launch_auto()
        self._set_status("READY")

    def _build_ui(self):
        status = tk.Frame(self, bg="#0a0c11", height=54)
        status.pack(fill=tk.X, side=tk.TOP)
        status.pack_propagate(False)
        tk.Label(status, text="⬡", fg=ACCENT_COLOR, bg="#0a0c11", font=F_XL_B).pack(side=tk.LEFT, padx=10)
        tk.Label(status, text="INDY ROS2", fg=ACCENT_COLOR, bg="#0a0c11", font=F_LG_B).pack(side=tk.LEFT)
        self.status_lbl = tk.Label(status, text="INIT", fg=DIM_COLOR, bg="#0a0c11", font=F_MD_B)
        self.status_lbl.pack(side=tk.LEFT, padx=20)
        self.clock_lbl = tk.Label(status, text="", fg=DIM_COLOR, bg="#0a0c11", font=F_MD)
        self.clock_lbl.pack(side=tk.RIGHT, padx=20)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left   = tk.Frame(paned, bg=PANEL_COLOR)
        center = tk.Frame(paned, bg=PANEL_COLOR)
        right  = tk.Frame(paned, bg=PANEL_COLOR)
        paned.add(left, weight=1)
        paned.add(center, weight=2)
        paned.add(right, weight=3)

        # controls
        tk.Label(left, text="CONTROLS", fg=DIM_COLOR, bg=PANEL_COLOR,
                 font=F_SM_B).pack(anchor='w', padx=10, pady=(10, 0))
        self.start_btn = tk.Button(left, text="▶  START", command=self._on_start,
                                   bg="#1a3a1f", fg=GREEN_COLOR, **BUTTON_STYLE)
        self.start_btn.pack(fill=tk.X, padx=10, pady=5)
        self.stop_btn = tk.Button(left, text="■  STOP", command=self._on_stop,
                                  bg="#3a1a1a", fg=RED_COLOR, **BUTTON_STYLE)
        self.stop_btn.pack(fill=tk.X, padx=10, pady=5)
        tk.Frame(left, height=8, bg=PANEL_COLOR).pack()

        tk.Label(left, text="CONFIGURATION", fg=DIM_COLOR, bg=PANEL_COLOR,
                 font=F_SM_B).pack(anchor='w', padx=10)
        for label, cmd in [("⚙  SETUP",       self._open_setup),
                            ("⚙  DEPTH SETUP", self._open_depth),
                            ("📄  MAIN LOG",    self._open_main_log)]:
            btn = tk.Button(left, text=label, command=cmd,
                            bg="#2a2010", fg=YELLOW_COLOR, **BUTTON_STYLE)
            btn.pack(fill=tk.X, padx=10, pady=3)
        tk.Frame(left, height=8, bg=PANEL_COLOR).pack()

        tk.Label(left, text="LOG", fg=DIM_COLOR, bg=PANEL_COLOR,
                 font=F_SM_B).pack(anchor='w', padx=10)
        tk.Button(left, text="🗑  CLEAR LOGS", command=self._clear_logs,
                  bg="#111c26", fg=ACCENT_COLOR, **BUTTON_STYLE
                  ).pack(fill=tk.X, padx=10, pady=3)
        tk.Button(left, text="📱  QR CODE",
                command=lambda: show_qr_window(self),
                bg="#0e1a1a", fg=ACCENT_COLOR, **BUTTON_STYLE
                ).pack(fill=tk.X, padx=10, pady=3)
        tk.Frame(left, height=8, bg=PANEL_COLOR).pack()

        tk.Label(left, text="SYSTEM STATUS", fg=DIM_COLOR, bg=PANEL_COLOR,
                 font=F_SM_B).pack(anchor='w', padx=10)
        sf = tk.Frame(left, bg="#0f1218")
        sf.pack(fill=tk.X, padx=10, pady=5)
        self.dot_ros  = tk.Label(sf, text="● ROS2 Node",   fg=DIM_COLOR, bg="#0f1218", font=F_MD)
        self.dot_ros.pack(anchor='w')
        self.dot_proc = tk.Label(sf, text="● Launch Proc", fg=DIM_COLOR, bg="#0f1218", font=F_MD)
        self.dot_proc.pack(anchor='w')
        self.dot_pub  = tk.Label(sf, text="● Publisher",   fg=DIM_COLOR, bg="#0f1218", font=F_MD)
        self.dot_pub.pack(anchor='w')

        # cameras
        tk.Label(center, text="CAMERAS", fg=DIM_COLOR, bg=PANEL_COLOR,
                 font=F_SM_B).pack(anchor='w', padx=10, pady=(10, 0))
        self.cam_detect = CameraWidget(center, "DETECT IMAGE", IMAGE_TOPIC_DETECT)
        self.cam_detect.pack(fill=tk.BOTH, expand=True, pady=5)
        self.cam_depth  = CameraWidget(center, "DEPTH IMAGE",  IMAGE_TOPIC_DEPTH)
        self.cam_depth.pack(fill=tk.BOTH, expand=True, pady=5)

        # logs
        tk.Label(right, text="LOGS", fg=DIM_COLOR, bg=PANEL_COLOR,
                 font=F_SM_B).pack(anchor='w', padx=10, pady=(10, 0))
        self.log_coords  = LogPanel(right, "COORDINATES", GREEN_COLOR)
        self.log_coords.pack(fill=tk.BOTH, expand=True, pady=2)
        self.log_collect = LogPanel(right, "COLLECT", YELLOW_COLOR)
        self.log_collect.pack(fill=tk.BOTH, expand=True, pady=2)

        self.log_main = LogPanel(None, "MAIN LOG", ACCENT_COLOR)
        self.main_log_window = None

    def _open_main_log(self):
        if self.main_log_window is not None and self.main_log_window.winfo_exists():
            self.main_log_window.lift()
            return
        self.main_log_window = tk.Toplevel(self)
        self.main_log_window.title("MAIN LOG VIEWER")
        self.main_log_window.attributes('-fullscreen', True)
        self.main_log_window.configure(bg=BG_COLOR)

        header = tk.Frame(self.main_log_window, bg="#0a0c11", height=54)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="📄  MAIN LOG", fg=ACCENT_COLOR, bg="#0a0c11",
                 font=F_LG_B).pack(side=tk.LEFT, padx=20)
        btn_close = tk.Button(header, text="✕  CLOSE",
                              command=self.main_log_window.destroy,
                              bg=RED_COLOR, fg="white", font=F_MD_B,
                              relief=tk.FLAT, padx=16, pady=8, cursor="hand2", bd=0)
        btn_close.pack(side=tk.RIGHT, padx=16, pady=10)
        btn_close.bind("<Enter>", lambda e: btn_close.config(bg="#cc2240"))
        btn_close.bind("<Leave>", lambda e: btn_close.config(bg=RED_COLOR))

        tk.Frame(self.main_log_window, bg=ACCENT_COLOR, height=1).pack(fill=tk.X)

        self.main_log_display = LogWidget(self.main_log_window, height=30)
        self.main_log_display.pack(fill=tk.BOTH, expand=True)

        footer = tk.Frame(self.main_log_window, bg="#0a0c11", height=50)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        tk.Frame(self.main_log_window, bg=DIM_COLOR, height=1).pack(fill=tk.X, side=tk.BOTTOM)

        def _clear_log_display():
            self.log_main.clear()
            self.main_log_display.config(state=tk.NORMAL)
            self.main_log_display.delete(1.0, tk.END)
            self.main_log_display.config(state=tk.DISABLED)

        btn_clear = tk.Button(footer, text="🗑  CLEAR LOG", command=_clear_log_display,
                              bg="#111c26", fg=ACCENT_COLOR, font=F_MD_B,
                              relief=tk.FLAT, padx=16, pady=6, cursor="hand2", bd=0)
        btn_clear.pack(side=tk.LEFT, padx=16, pady=10)
        btn_clear.bind("<Enter>", lambda e: btn_clear.config(bg="#1a2c40"))
        btn_clear.bind("<Leave>", lambda e: btn_clear.config(bg="#111c26"))

        self._sync_main_log()

    def _sync_main_log(self):
        if self.main_log_window and self.main_log_window.winfo_exists():
            text = self.log_main.log.get(1.0, tk.END)
            self.main_log_display.config(state=tk.NORMAL)
            self.main_log_display.delete(1.0, tk.END)
            self.main_log_display.insert(tk.END, text)
            self.main_log_display.see(tk.END)
            self.main_log_display.config(state=tk.DISABLED)
            self.after(500, self._sync_main_log)

    def _clock_timer(self):
        self.clock_lbl.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._clock_timer)

    def _drain_timer(self):
        for _ in range(15):
            try:
                line = self._proc_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                break
            low = line.lower()
            if COORD_KW.lower() in low:
                self.log_coords.append(line)
            elif COLLECT_KW.lower() in low:
                self.log_collect.append(line)
            else:
                self.log_main.append(line)
        self.after(150, self._drain_timer)

    def _set_status(self, text, color=None):
        self.status_lbl.config(text=text, fg=color or DIM_COLOR)

    def _init_ros(self):
        if not HAS_ROS:
            self.log_main.append("[HMI] rclpy not found — ROS features disabled")
            self.dot_ros.config(fg=RED_COLOR)
            return
        try:
            rclpy.init(args=None)
            self._ros_node = rclpy.create_node("hmi_node")
            self._start_pub = self._ros_node.create_publisher(StartMsg, START_TOPIC, 10)
            executor = SingleThreadedExecutor()
            executor.add_node(self._ros_node)
            self._ros_worker = RosWorker(self._ros_node, executor)
            self._ros_node.create_subscription(
                RosImage, IMAGE_TOPIC_DETECT,
                lambda msg: self._on_image(IMAGE_TOPIC_DETECT, msg), 1)
            self._ros_node.create_subscription(
                RosImage, IMAGE_TOPIC_DEPTH,
                lambda msg: self._on_image(IMAGE_TOPIC_DEPTH, msg), 1)
            self._ros_thread = threading.Thread(target=self._ros_worker.spin, daemon=True)
            self._ros_thread.start()
            self.dot_ros.config(fg=GREEN_COLOR)
            self.dot_pub.config(fg=GREEN_COLOR)
            self.log_main.append("[HMI] ROS2 node ready")
        except Exception as e:
            self.dot_ros.config(fg=RED_COLOR)
            self.log_main.append(f"[HMI] ROS2 init error: {e}")

    def _on_image(self, topic, msg):
        if not self._ros_worker:
            return
        arr = self._ros_worker.decode_image(topic, msg)
        if arr is not None:
            self.after(0, lambda t=topic, a=arr: self._update_camera(t, a))

    def _update_camera(self, topic, arr):
        if topic == IMAGE_TOPIC_DETECT:
            self.cam_detect.update_image(arr)
        elif topic == IMAGE_TOPIC_DEPTH:
            self.cam_depth.update_image(arr)

    def _pub_start(self, value):
        if not HAS_ROS or self._start_pub is None:
            self.log_main.append(f"[HMI] ROS unavailable — cannot publish start={value}")
            return
        try:
            msg = StartMsg()
            msg.start = value
            self._start_pub.publish(msg)
            self.log_main.append(f"[HMI] Published start={value} → {START_TOPIC}")
        except Exception as e:
            self.log_main.append(f"[HMI] Publish error: {e}")

    def _launch_auto(self):
        if not AUTO_CMD.strip():
            self.log_main.append("[HMI] AUTO_CMD empty, skipping.")
            self.dot_proc.config(fg=DIM_COLOR)
            return
        try:
            self._proc_running = True
            self._proc = subprocess.Popen(
                AUTO_CMD, shell=True, executable="/bin/bash",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, preexec_fn=os.setsid)
            self._proc_thread = threading.Thread(target=self._read_proc, daemon=True)
            self._proc_thread.start()
            self.dot_proc.config(fg=GREEN_COLOR)
            self.log_main.append(f"[HMI] Auto-launch started — PID {self._proc.pid}")
            self._set_status("RUNNING", GREEN_COLOR)
        except Exception as e:
            self.dot_proc.config(fg=RED_COLOR)
            self.log_main.append(f"[HMI] Auto-launch error: {e}")
            self._set_status("ERROR", RED_COLOR)

    def _read_proc(self):
        try:
            while self._proc_running and self._proc:
                line = self._proc.stdout.readline()
                if not line:
                    break
                if len(line) > 400:
                    line = line[:400] + " ..."
                self._proc_queue.put(line)
        except Exception:
            pass
        finally:
            self._proc_queue.put(None)

    def _kill_auto(self):
        """Kill toàn bộ cây process của AUTO_CMD (kể cả Gazebo, ROS2 nodes con).

        ros2 launch tạo nhiều child/grandchild ở process group khác nhau,
        nên killpg không đủ. Dùng psutil để walk toàn bộ cây PID.
        """
        self._proc_running = False
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        def _collect_tree(pid):
            """Thu thập PID gốc + tất cả con cháu (psutil)."""
            try:
                import psutil
                root = psutil.Process(pid)
                children = root.children(recursive=True)
                return children + [root]
            except Exception:
                return []

        def _do_kill():
            # 1. Thu thập cây TRƯỚC khi gửi signal (sau SIGTERM cây có thể tan)
            tree = _collect_tree(proc.pid)

            # 2. SIGTERM toàn bộ cây
            for p in tree:
                try:
                    p.send_signal(_signal.SIGTERM)
                except Exception:
                    pass

            # Fallback: killpg cho process group của shell
            try:
                os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
            except OSError:
                pass

            # 3. Chờ tối đa 2 giây
            import psutil
            gone, alive = psutil.wait_procs(tree, timeout=2.0)

            # 4. Những gì còn sống → SIGKILL
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
            except OSError:
                pass

            # 5. Đóng stdout pipe
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

            # 6. Reap zombie
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass

        t = threading.Thread(target=_do_kill, daemon=True, name="proc-killer")
        t.start()
        t.join(timeout=4.0)   # tổng: 2s wait + 1s reap + buffer

        try:
            self.dot_proc.config(fg=DIM_COLOR)
        except Exception:
            pass

    def _on_start(self):
        self._pub_start(True)
        self._set_status("STARTED", GREEN_COLOR)

    def _on_stop(self):
        self._pub_start(False)
        self._set_status("STOPPED", RED_COLOR)

    def _clear_logs(self):
        self.log_main.clear()
        self.log_coords.clear()
        self.log_collect.clear()
        if self.main_log_window and self.main_log_window.winfo_exists():
            self.main_log_display.config(state=tk.NORMAL)
            self.main_log_display.delete(1.0, tk.END)
            self.main_log_display.config(state=tk.DISABLED)

    def _open_setup(self):
        SetupDialog(self, SETUP_PATH, SETUP_LABELS, yaml_section="setup").focus_set()

    def _open_depth(self):
        DepthSetupDialog(self).focus_set()

    def _shutdown(self):
        self._proc_running = False
        if self._ros_worker:
            self._ros_worker.stop()
        if self._ros_node:
            try:
                self._ros_node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
        self._kill_auto()
        self.quit()

    def closeEvent(self):
        self._shutdown()
        try:
            self.destroy()
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────
def sigint_handler(signum, frame):
    print("\nCtrl+C — shutting down...")
    if 'app' in globals() and app:
        try:
            app._shutdown()   # kill ROS + proc, rồi quit mainloop
        except Exception:
            pass
        try:
            app.destroy()
        except Exception:
            pass
    os._exit(0)

def main():
    global app
    _signal.signal(_signal.SIGINT, sigint_handler)
    app = MainWindow()
    app.protocol("WM_DELETE_WINDOW", app.closeEvent)
    app.mainloop()

if __name__ == "__main__":
    main()