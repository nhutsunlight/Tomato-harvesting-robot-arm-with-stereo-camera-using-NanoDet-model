#!/usr/bin/env python3
"""
Indy ROS2 HMI — Tkinter (optimised)
Entry point: main()
"""

import sys
import signal as _signal
import threading
import queue
import time
import yaml
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk

# ── Optional ROS2 ─────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image as RosImage
    from start_msgs.msg import StartMsg
    from rcl_interfaces.msg import Log as RosLog
    import numpy as np
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    np = None

try:
    from hmi_launcher.hmi_qr import show_qr_window
except ImportError:
    show_qr_window = None

try:
    from hmi_launcher.log_bus import bus as _log_bus
except ImportError:
    _log_bus = None

# ─── Config paths ─────────────────────────────────────────────────────────────
def _find_config() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        c = parent / 'src' / 'indy-ros2' / 'msg' / 'config_manager' / 'config'
        if c.exists():
            return c
    return Path("/home/nhut/indy-ros2/config")

_CFG              = _find_config()
SETUP_PATH        = _CFG / 'setup.yaml'
DEPTH_CONFIG_PATH = _CFG / 'stereo_config.yaml'

IMAGE_TOPIC_DETECT = "/stereo/left/image_yolo"
IMAGE_TOPIC_DEPTH  = "/stereo/depth/image_raw"
START_TOPIC        = "/start_msg"
COORD_KW           = "Updated target:"
COLLECT_KW         = "collect_logger_node"
LOG_MAX_LINES      = 400
FULLSCREEN         = False

# ── Config labels ─────────────────────────────────────────────────────────────
SETUP_LABELS = {
    "HomePose":           "Home Pose (joint list)",
    "DropPose":           "Drop Pose (joint list)",
    "OffSetDistance":     "Offset Distance (m)",
    "YOffSetDistance":    "Y Offset Distance (m)",
    "OffSetAngle":        "Offset Angle (rad)",
    "FxOffset":           "Fx Offset (m)",
    "ObjectOffset":       "Object Offset (m)",
    "Multi_collect_mode": "Multi Collect Mode (bool)",
}
STEREO_BM_LABELS = {
    "numDisparities": "Num Disparities",    "blockSize":        "Block Size (odd≥1)",
    "preFilterType":  "Pre-Filter Type",    "preFilterSize":    "Pre-Filter Size",
    "preFilterCap":   "Pre-Filter Cap",     "textureThreshold": "Texture Threshold",
    "uniquenessRatio":"Uniqueness Ratio (%)", "speckleWindowSize":"Speckle Window",
    "speckleRange":   "Speckle Range",      "disp12MaxDiff":    "Disp12 Max Diff",
}
STEREO_SGBM_LABELS = {
    "minDisparity":   "Min Disparity",      "numDisparities":   "Num Disparities",
    "blockSize":      "Block Size (odd≥1)", "P1":               "P1 (SGBM penalty)",
    "P2":             "P2 (SGBM penalty)",  "disp12MaxDiff":    "Disp12 Max Diff",
    "uniquenessRatio":"Uniqueness Ratio (%)", "speckleWindowSize":"Speckle Window",
    "speckleRange":   "Speckle Range",      "preFilterCap":     "Pre-Filter Cap",
    "mode":           "Algorithm Mode",
}

# ══════════════════════════════════════════════════════════════════════════════
# ── STYLE
# ══════════════════════════════════════════════════════════════════════════════
BG_COLOR     = "#0d0f14"
PANEL_COLOR  = "#131720"
ACCENT_COLOR = "#00e5ff"
GREEN_COLOR  = "#39ff7e"
RED_COLOR    = "#ff3b5c"
YELLOW_COLOR = "#ffd166"
TEXT_COLOR   = "#c8d6e5"
DIM_COLOR    = "#4a5568"
FONT         = "Courier New"

F_XS  = (FONT,10); F_XS_B = (FONT,10,"bold")
F_SM  = (FONT,12); F_SM_B = (FONT,12,"bold")
F_MD  = (FONT,14); F_MD_B = (FONT,14,"bold")
F_LG  = (FONT,16); F_LG_B = (FONT,16,"bold")
F_XL  = (FONT,20); F_XL_B = (FONT,20,"bold")

SETUP_ROW_FONT  = F_LG_B; SETUP_ROW_KEY  = F_MD;   SETUP_ROW_VAL  = F_LG_B
SETUP_ROW_IDX   = F_MD;   SETUP_ROW_BADGE= F_SM_B; SETUP_ROW_PAD_Y = 12
SETUP_HDR       = F_LG_B; SETUP_ED_TITLE = F_LG_B; SETUP_ED_KEY   = F_MD
SETUP_ED_CURVAL = F_LG_B; SETUP_ED_INPUT = F_LG_B; SETUP_ED_BTN   = F_MD_B
SETUP_ED_LBL    = F_MD;   SETUP_KBD      = F_LG_B; SETUP_KBD_SP   = F_MD_B

BTN = {"font": F_LG_B, "relief": tk.FLAT, "bd": 0, "padx": 10, "pady": 8, "cursor": "hand2"}

# ══════════════════════════════════════════════════════════════════════════════
# ── Log Widget
# ══════════════════════════════════════════════════════════════════════════════
class LogWidget(scrolledtext.ScrolledText):
    def __init__(self, parent, **kw):
        super().__init__(parent, wrap=tk.WORD, font=F_MD,
                         bg=BG_COLOR, fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                         relief=tk.FLAT, bd=0, **kw)
        self.config(state=tk.DISABLED)
        self._q  = queue.Queue()
        self._aid = None

    def append_safe(self, text: str):
        self._q.put(f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
        if self._aid is None:
            self._aid = self.after(50, self._flush)

    def _flush(self):
        self._aid = None
        self.config(state=tk.NORMAL)
        n = 0
        while not self._q.empty() and n < 30:
            self.insert(tk.END, self._q.get()); n += 1
        lines = int(self.index('end-1c').split('.')[0])
        if lines > LOG_MAX_LINES:
            self.delete('1.0', f'{lines - LOG_MAX_LINES}.0')
        self.see(tk.END)
        self.config(state=tk.DISABLED)
        if not self._q.empty():
            self._aid = self.after(50, self._flush)

    def clear(self):
        self.config(state=tk.NORMAL); self.delete('1.0', tk.END); self.config(state=tk.DISABLED)


class LogPanel(tk.Frame):
    _CH_MAP = {'COORDINATES': 'log_coords', 'COLLECT': 'log_collect'}

    def __init__(self, parent, title, color):
        super().__init__(parent, bg=PANEL_COLOR, bd=0)
        self._count = 0
        self._title = title.upper()
        hdr = tk.Frame(self, bg="#0a0c11", height=34)
        hdr.pack(fill=tk.X, side=tk.TOP); hdr.pack_propagate(False)
        tk.Label(hdr, text="●", fg=color, bg="#0a0c11", font=F_SM).pack(side=tk.LEFT, padx=(8,4))
        tk.Label(hdr, text=self._title, fg=color, bg="#0a0c11", font=F_SM_B).pack(side=tk.LEFT)
        self._cnt_lbl = tk.Label(hdr, text="0", fg=DIM_COLOR, bg="#1e2535", font=F_SM, padx=5, pady=1)
        self._cnt_lbl.pack(side=tk.RIGHT, padx=8)
        self.log = LogWidget(self, height=12)
        self.log.pack(fill=tk.BOTH, expand=True)

    def append(self, line: str):
        self._count += 1
        self._cnt_lbl.config(text=str(self._count))
        self.log.append_safe(line)
        if _log_bus is not None:
            try: _log_bus.push(self._CH_MAP.get(self._title, 'log_main'), line)
            except Exception: pass

    def clear(self):
        self._count = 0; self._cnt_lbl.config(text="0"); self.log.clear()


# ══════════════════════════════════════════════════════════════════════════════
# ── Camera Widget — update only when new frame arrives
# ══════════════════════════════════════════════════════════════════════════════
class CameraWidget(tk.Frame):
    def __init__(self, parent, title, topic):
        super().__init__(parent, bg=PANEL_COLOR)
        self.topic = topic
        self._last_t = None
        self._lock   = threading.Lock()
        self._arr    = None      # latest decoded array waiting to render
        self._dirty  = False     # True = new frame not yet rendered

        hdr = tk.Frame(self, bg="#0a0c11", height=28)
        hdr.pack(fill=tk.X, side=tk.TOP); hdr.pack_propagate(False)
        tk.Label(hdr, text="●", fg=ACCENT_COLOR, bg="#0a0c11", font=F_XS).pack(side=tk.LEFT, padx=(8,4))
        tk.Label(hdr, text=title, fg=ACCENT_COLOR, bg="#0a0c11", font=F_SM_B).pack(side=tk.LEFT)
        self._fps = tk.Label(hdr, text="no signal", fg=DIM_COLOR, bg="#0a0c11", font=F_SM)
        self._fps.pack(side=tk.RIGHT, padx=8)
        self.canvas = tk.Canvas(self, bg="#050608", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._photo = None
        self._img_item = None

    def post_array(self, arr):
        """Called from decode thread. Store array and schedule exactly one render."""
        with self._lock:
            self._arr   = arr
            was_dirty   = self._dirty
            self._dirty = True
        if not was_dirty:
            # Only schedule when transitioning clean→dirty
            self.after(0, self._render)

    def _render(self):
        """Runs on Tk main thread. Renders the latest stored array."""
        with self._lock:
            arr         = self._arr
            self._arr   = None
            self._dirty = False   # mark clean AFTER taking the array

        if arr is None:
            return
        try:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w < 10 or h < 10:
                w, h = 320, 240
            img = Image.fromarray(arr).resize((w, h), Image.Resampling.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
            if self._img_item is None:
                self._img_item = self.canvas.create_image(w // 2, h // 2, anchor=tk.CENTER, image=self._photo)
            else:
                self.canvas.coords(self._img_item, w // 2, h // 2)
                self.canvas.itemconfig(self._img_item, image=self._photo)
            now = time.time()
            if self._last_t:
                self._fps.config(text=f"{1/(now - self._last_t):.1f} fps")
            self._last_t = now
        except Exception as e:
            print(f"[CAM] render error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ── ROS Worker — decode in thread pool, no throttle (update on every msg)
# ══════════════════════════════════════════════════════════════════════════════
class RosWorker:
    def __init__(self, node, executor):
        self.node     = node
        self.executor = executor
        self.running  = True
        from concurrent.futures import ThreadPoolExecutor
        # 1 worker per camera = 2 total; keeps CPU usage bounded
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cam_dec")
        self._decode_lock = threading.Lock()
        self._decode_pending = {}

    def spin(self):
        while self.running:
            try: self.executor.spin_once(timeout_sec=0.05)
            except Exception: break

    def stop(self):
        self.running = False
        try: self._pool.shutdown(wait=False)
        except Exception: pass
        try: self.executor.shutdown()
        except Exception: pass

    def submit_decode(self, topic: str, msg, widget: CameraWidget):
        """
        ROS callback: decode newest frame only.
        Prevent unbounded backlog when camera publish rate is higher than decode/render speed.
        """
        with self._decode_lock:
            if self._decode_pending.get(topic, False):
                return
            self._decode_pending[topic] = True
        self._pool.submit(self._decode_wrapped, topic, msg, widget)

    def _decode_wrapped(self, topic: str, msg, widget: CameraWidget):
        try:
            RosWorker._decode(topic, msg, widget)
        finally:
            with self._decode_lock:
                self._decode_pending[topic] = False

    @staticmethod
    def _decode(topic: str, msg, widget: CameraWidget):
        try:
            enc = msg.encoding.lower()
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if enc == "mono8":
                arr = np.stack([raw.reshape(msg.height, msg.width)] * 3, axis=2)
            elif enc == "rgb8":
                arr = raw.reshape(msg.height, msg.width, 3)
            elif enc == "bgr8":
                arr = raw.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
            else:
                arr = raw.reshape(msg.height, msg.width, -1)[:, :, :3].copy()
            widget.post_array(arr)
        except Exception as e:
            print(f"[CAM] decode error {topic}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ── Setup Dialog
# ══════════════════════════════════════════════════════════════════════════════
class SetupDialog(tk.Toplevel):
    _C = {
        "bg":"#0b0d12", "sidebar":"#0e1118", "card":"#161b26", "card_sel":"#1a2235",
        "border":"#1e2840", "border_hi":"#00e5ff", "accent":"#00e5ff", "green":"#39ff7e",
        "red":"#ff3b5c", "yellow":"#ffd166", "dim":"#3d4f68", "text":"#c8d6e5",
        "subtext":"#7a8fa8", "kbd_bg":"#0e1118", "kbd_key":"#1c2333",
        "kbd_key_h":"#00e5ff", "editor_bg":"#10131a",
    }
    KBD_H = 320

    def __init__(self, parent, path, labels, yaml_section=None):
        super().__init__(parent)
        self.title(f"SETUP — {path.name}")
        self.attributes('-fullscreen', FULLSCREEN)
        self.configure(bg=self._C["bg"])
        self.path = path; self.labels = labels; self.section = yaml_section
        self.data = {}; self.current_key = None
        self._edit_mode = False; self._cards = {}
        self._load(); self._build()

    def _load(self):
        try:
            raw = yaml.safe_load(self.path.read_text()) or {}
            src = raw.get(self.section, raw) if self.section else raw
            self.data = dict(src)
        except Exception as e: messagebox.showerror("Load Error", str(e))

    def _save(self):
        try:
            raw = yaml.safe_load(self.path.read_text()) or {}
            sec = raw.setdefault(self.section, {}) if self.section else raw
            sec.update(self.data)
            with open(self.path, 'w') as f:
                yaml.dump(raw, f, allow_unicode=True, sort_keys=False)
            messagebox.showinfo("Saved", "Configuration saved.")
        except Exception as e: messagebox.showerror("Save Error", str(e))

    @staticmethod
    def _fv(v):
        if isinstance(v, list):
            s = "[" + ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v) + "]"
            return s if len(s) <= 32 else s[:29] + "…]"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    @staticmethod
    def _fv_full(v):
        if isinstance(v, list):
            return "[" + ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v) + "]"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def _build(self):
        C = self._C
        hdr = tk.Frame(self, bg="#080a0f", height=56); hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        self._build_header(hdr)
        tk.Frame(self, bg=C["border_hi"], height=2).pack(fill=tk.X)
        self._top = tk.Frame(self, bg=C["sidebar"]); self._top.pack(fill=tk.BOTH, expand=True)
        self._build_param_list()
        self._bot = tk.Frame(self, bg=C["editor_bg"])

    def _build_header(self, hdr):
        C = self._C
        tk.Label(hdr, text="⚙", fg=C["accent"], bg="#080a0f", font=F_XL_B).pack(side=tk.LEFT, padx=(20,6), pady=8)
        tk.Label(hdr, text=self.path.name.upper(), fg=C["accent"], bg="#080a0f", font=F_LG_B).pack(side=tk.LEFT, pady=8)
        tk.Label(hdr, text=f"  •  {len(self.labels)} parameters", fg=C["dim"], bg="#080a0f", font=F_SM).pack(side=tk.LEFT, pady=8)
        b = tk.Button(hdr, text="✕  CLOSE", command=self.destroy, bg=C["red"], fg="white",
                      font=F_MD_B, relief=tk.FLAT, padx=18, pady=10, cursor="hand2", bd=0)
        b.pack(side=tk.RIGHT, padx=20, pady=10)
        b.bind("<Enter>", lambda e: b.config(bg="#cc2240"))
        b.bind("<Leave>", lambda e: b.config(bg=C["red"]))

    def _build_param_list(self):
        C = self._C; ctr = self._top
        sh = tk.Frame(ctr, bg=C["sidebar"], height=36); sh.pack(fill=tk.X); sh.pack_propagate(False)
        tk.Label(sh, text="  SELECT PARAMETER", fg=C["dim"], bg=C["sidebar"], font=SETUP_HDR).pack(side=tk.LEFT, padx=12, pady=6)
        tk.Label(sh, text=str(len(self.labels)), fg=C["accent"], bg=C["sidebar"], font=SETUP_HDR).pack(side=tk.RIGHT, padx=12)
        tk.Frame(ctr, bg=C["border"], height=1).pack(fill=tk.X)

        vsb = tk.Scrollbar(ctr, orient=tk.VERTICAL); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        vc  = tk.Canvas(ctr, bg=C["sidebar"], highlightthickness=0, bd=0, yscrollcommand=vsb.set)
        vc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb.configure(command=vc.yview)
        rf  = tk.Frame(vc, bg=C["sidebar"]); win = vc.create_window((0,0), window=rf, anchor="nw")
        rf.bind("<Configure>", lambda e: vc.configure(scrollregion=vc.bbox("all")))
        vc.bind("<Configure>", lambda e: vc.itemconfig(win, width=e.width))
        for w in (vc, rf):
            w.bind("<MouseWheel>", lambda e: vc.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._cards = {}
        for i, (k, lbl) in enumerate(self.labels.items()):
            self._make_row(rf, k, lbl, i)

    def _make_row(self, parent, key, label, idx):
        C = self._C; val = self.data.get(key)
        if isinstance(val, bool):   btext,bbg,tcol = "BOOL","#ffd166",C["yellow"]
        elif isinstance(val, list): btext,bbg,tcol = "LIST","#39ff7e",C["green"]
        else:                       btext,bbg,tcol = "NUM", "#00e5ff",C["accent"]

        row = tk.Frame(parent, bg=C["card"], cursor="hand2", bd=0); row.pack(fill=tk.X)
        bar = tk.Frame(row, bg=C["card"], width=4); bar.pack(side=tk.LEFT, fill=tk.Y); bar.pack_propagate(False)
        inn = tk.Frame(row, bg=C["card"], padx=14, pady=SETUP_ROW_PAD_Y); inn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tl  = tk.Frame(inn, bg=C["card"]); tl.pack(fill=tk.X)
        tk.Label(tl, text=f"{idx+1:02d}", fg=C["dim"], bg=C["card"], font=SETUP_ROW_IDX).pack(side=tk.LEFT, padx=(0,8))
        tk.Label(tl, text=label, fg=C["text"], bg=C["card"], font=SETUP_ROW_FONT, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(tl, text=btext, fg="#0d0f14", bg=bbg, font=SETUP_ROW_BADGE, padx=6, pady=2).pack(side=tk.RIGHT)
        bl  = tk.Frame(inn, bg=C["card"]); bl.pack(fill=tk.X, pady=(4,0))
        tk.Label(bl, text=key, fg=C["dim"], bg=C["card"], font=SETUP_ROW_KEY).pack(side=tk.LEFT)
        vl  = tk.Label(bl, text=self._fv(val), fg=tcol, bg=C["card"], font=SETUP_ROW_VAL); vl.pack(side=tk.RIGHT)
        tk.Frame(parent, bg=C["border"], height=1).pack(fill=tk.X)

        widgets = [row, inn, tl, bl] + list(tl.winfo_children()) + list(bl.winfo_children())
        def _on_enter(e, r=row, a=bar):
            for w in [r,inn,tl,bl]: w.config(bg=C["card_sel"])
            a.config(bg=C["accent"])
        def _on_leave(e, r=row, a=bar, k=key):
            bg = C["card_sel"] if self.current_key==k else C["card"]
            ac = C["accent"]   if self.current_key==k else C["card"]
            for w in [r,inn,tl,bl]: w.config(bg=bg)
            a.config(bg=ac)
        for w in widgets:
            try:
                w.bind("<Enter>",    _on_enter)
                w.bind("<Leave>",    _on_leave)
                w.bind("<Button-1>", lambda e, k=key: self._select(k))
            except Exception: pass
        self._cards[key] = {"row":row,"bar":bar,"inn":inn,"tl":tl,"bl":bl,"vl":vl}

    def _select(self, key):
        C = self._C
        if self.current_key and self.current_key in self._cards:
            p = self._cards[self.current_key]
            for w in [p["row"],p["inn"],p["tl"],p["bl"]]: w.config(bg=C["card"])
            p["bar"].config(bg=C["card"])
        self.current_key = key
        cur = self._cards[key]
        for w in [cur["row"],cur["inn"],cur["tl"],cur["bl"]]: w.config(bg=C["card_sel"])
        cur["bar"].config(bg=C["accent"])
        self._enter_edit(key)

    def _update_val(self, key, s):
        if key in self._cards: self._cards[key]["vl"].config(text=s)

    def _enter_edit(self, key):
        if self._edit_mode: self._show_editor(key); return
        self._edit_mode = True
        self._top.pack_forget(); self._bot.pack(fill=tk.BOTH, expand=True)
        self._show_editor(key)

    def _exit_edit(self):
        self._edit_mode = False
        self._bot.pack_forget(); self._top.pack(fill=tk.BOTH, expand=True)
        if self.current_key and self.current_key in self._cards:
            p = self._cards[self.current_key]
            for w in [p["row"],p["inn"],p["tl"],p["bl"]]: w.config(bg=self._C["card"])
            p["bar"].config(bg=self._C["card"])
        self.current_key = None

    def _show_editor(self, key):
        C = self._C
        for w in self._bot.winfo_children(): w.destroy()
        val = self.data.get(key); label = self.labels.get(key, key)
        has_kbd = not isinstance(val, bool)

        br = tk.Frame(self._bot, bg=C["editor_bg"]); br.pack(fill=tk.X, padx=20, pady=(12,0))
        bb = tk.Button(br, text="←  BACK", command=self._exit_edit,
                       bg=C["border"], fg=C["text"], font=F_MD_B, relief=tk.FLAT, padx=14, pady=8, cursor="hand2", bd=0)
        bb.pack(side=tk.LEFT)
        bb.bind("<Enter>", lambda e: bb.config(bg=C["dim"]))
        bb.bind("<Leave>", lambda e: bb.config(bg=C["border"]))

        entry_ref = [None]
        if has_kbd:
            tk.Frame(self._bot, bg=C["border"], height=1).pack(side=tk.BOTTOM, fill=tk.X)
            kbf = tk.Frame(self._bot, bg=C["kbd_bg"], height=self.KBD_H)
            kbf.pack(side=tk.BOTTOM, fill=tk.X); kbf.pack_propagate(False)
            self._kbf = kbf

        form = tk.Frame(self._bot, bg=C["editor_bg"]); form.pack(fill=tk.BOTH, expand=True, padx=30, pady=14)
        nr = tk.Frame(form, bg=C["editor_bg"]); nr.pack(fill=tk.X, pady=(0,10))
        tk.Label(nr, text=label, fg=C["accent"], bg=C["editor_bg"], font=SETUP_ED_TITLE).pack(side=tk.LEFT)
        tk.Label(nr, text=f"  ·  {key}", fg=C["dim"], bg=C["editor_bg"], font=SETUP_ED_KEY).pack(side=tk.LEFT, pady=(4,0))

        pill = tk.Frame(form, bg=C["card"]); pill.pack(fill=tk.X, pady=(0,14))
        tk.Frame(pill, bg=C["accent"], width=4).pack(side=tk.LEFT, fill=tk.Y)
        pi = tk.Frame(pill, bg=C["card"], padx=16, pady=12); pi.pack(fill=tk.X, expand=True)
        tk.Label(pi, text="CURRENT VALUE", fg=C["dim"], bg=C["card"], font=F_XS_B).pack(anchor="w")
        tk.Label(pi, text=self._fv_full(val), fg=C["yellow"], bg=C["card"],
                 font=SETUP_ED_CURVAL, wraplength=900, justify=tk.LEFT).pack(anchor="w", pady=(4,0))
        tk.Frame(form, bg=C["border"], height=1).pack(fill=tk.X, pady=(0,14))

        if isinstance(val, bool):   self._bool_editor(form, key, val, C)
        elif isinstance(val, list): entry_ref[0] = self._list_editor(form, key, val, C)
        else:                       entry_ref[0] = self._scalar_editor(form, key, val, C)

        if has_kbd and entry_ref[0]: self._build_kbd(self._kbf, entry_ref[0], C)

    def _bool_editor(self, parent, key, val, C):
        var = tk.BooleanVar(value=val)
        f = tk.Frame(parent, bg=C["editor_bg"]); f.pack(fill=tk.X)
        tk.Label(f, text="Toggle value:", fg=C["subtext"], bg=C["editor_bg"], font=SETUP_ED_LBL).pack(anchor="w", pady=(0,12))
        br = tk.Frame(f, bg=C["editor_bg"]); br.pack(anchor="w")
        for txt, bv, col in [("●  ON",True,C["green"]),("●  OFF",False,C["red"])]:
            rb = tk.Radiobutton(br, text=txt, variable=var, value=bv, bg=C["card"], fg=col,
                                selectcolor=C["card_sel"], activebackground=C["card"], activeforeground=col,
                                font=SETUP_ED_BTN, relief=tk.FLAT, padx=26, pady=16, cursor="hand2", indicatoron=0, width=10)
            rb.pack(side=tk.LEFT, padx=(0,14))
        def sv(): self.data[key]=var.get(); self._save(); self._update_val(key,self._fv(self.data[key]))
        self._save_btn(parent, sv, C)

    def _list_editor(self, parent, key, val, C):
        f = tk.Frame(parent, bg=C["editor_bg"]); f.pack(fill=tk.X)
        tk.Label(f, text="Edit list (comma-separated):", fg=C["subtext"], bg=C["editor_bg"], font=SETUP_ED_LBL).pack(anchor="w", pady=(0,10))
        ef = tk.Frame(f, bg=C["accent"], padx=2, pady=2); ef.pack(fill=tk.X)
        e = tk.Entry(ef, font=SETUP_ED_INPUT, bg=C["card"], fg=C["text"], insertbackground=C["accent"], relief=tk.FLAT, bd=10)
        e.pack(fill=tk.X); e.insert(0, ", ".join(str(x) for x in val))
        def sv():
            try:
                nv = [float(p.strip()) if '.' in p else int(p.strip()) for p in e.get().split(',')]
                self.data[key]=nv; self._save(); self._update_val(key, self._fv(nv))
            except Exception: messagebox.showerror("Error","Invalid list format")
        self._save_btn(parent, sv, C); return e

    def _scalar_editor(self, parent, key, val, C):
        f = tk.Frame(parent, bg=C["editor_bg"]); f.pack(fill=tk.X)
        tk.Label(f, text="New value:", fg=C["subtext"], bg=C["editor_bg"], font=SETUP_ED_LBL).pack(anchor="w", pady=(0,10))
        ef = tk.Frame(f, bg=C["accent"], padx=2, pady=2); ef.pack(fill=tk.X)
        e = tk.Entry(ef, font=SETUP_ED_INPUT, bg=C["card"], fg=C["accent"], insertbackground=C["accent"], relief=tk.FLAT, bd=10)
        e.pack(fill=tk.X); e.insert(0, str(val)); e.icursor(tk.END); e.focus_set()
        def sv():
            try:
                if isinstance(val,int) and not isinstance(val,bool): nv=int(e.get())
                elif isinstance(val,float): nv=float(e.get())
                else: nv=e.get()
                self.data[key]=nv; self._save(); self._update_val(key, self._fv(nv))
            except Exception: messagebox.showerror("Error",f"Expected {type(val).__name__}")
        self._save_btn(parent, sv, C); return e

    def _save_btn(self, parent, cmd, C):
        f = tk.Frame(parent, bg=C["editor_bg"]); f.pack(fill=tk.X, pady=(16,0))
        b = tk.Button(f, text="💾  SAVE CHANGES", command=cmd, bg=C["green"], fg="#0b0d12",
                      font=SETUP_ED_BTN, relief=tk.FLAT, padx=26, pady=12, cursor="hand2", bd=0)
        b.pack(side=tk.LEFT)
        b.bind("<Enter>", lambda e: b.config(bg="#2be065"))
        b.bind("<Leave>", lambda e: b.config(bg=C["green"]))

    def _build_kbd(self, parent, target, C):
        ROWS = ["`1234567890-=","qwertyuiop[]\\","asdfghjkl;'","zxcvbnm,./"]
        SH   = {"`":"~","1":"!","2":"@","3":"#","4":"$","5":"%","6":"^","7":"&",
                "8":"*","9":"(","0":")","-":"_","=":"+","[":"{","]":"}","\\":"|",
                ";":":","'":'"',",":"<",".":">","/":"?"}
        ss=[False]; cs=[False]; btns=[]
        w = tk.Frame(parent, bg=C["kbd_bg"], padx=16, pady=10); w.pack(fill=tk.BOTH, expand=True)
        ho = lambda b: b.config(bg=C["kbd_key_h"], fg="#0b0d12")
        hf = lambda b: b.config(bg=C["kbd_key"],   fg=C["text"])
        def upd():
            a=ss[0]or cs[0]
            for ch,b in btns:
                if len(ch)==1: b.config(text=SH.get(ch,ch.upper()) if a else ch)
        def key(ch):
            a=ss[0]or cs[0]; out=SH.get(ch,ch.upper()) if a else ch
            target.insert(tk.INSERT,out); target.focus_set()
            if ss[0] and not cs[0]: ss[0]=False; upd()
        for row in ROWS:
            rf=tk.Frame(w,bg=C["kbd_bg"]); rf.pack(pady=2)
            for ch in row:
                b=tk.Button(rf,text=ch,width=3,height=1,font=SETUP_KBD,bg=C["kbd_key"],fg=C["text"],
                             relief=tk.FLAT,bd=0,cursor="hand2",command=lambda c=ch:key(c))
                b.pack(side=tk.LEFT,padx=2)
                b.bind("<Enter>",lambda e,btn=b:ho(btn)); b.bind("<Leave>",lambda e,btn=b:hf(btn))
                btns.append((ch,b))
        sp=tk.Frame(w,bg=C["kbd_bg"]); sp.pack(pady=(6,0))
        def _bs():
            p=target.index(tk.INSERT)
            if p!="0": target.delete(f"{p} -1c")
        for lbl,wd,cmd in [("SHIFT",5,lambda:(ss.__setitem__(0,not ss[0]),upd())),
                            ("CAPS",5, lambda:(cs.__setitem__(0,not cs[0]),upd())),
                            ("SPACE",16,lambda:(target.insert(tk.INSERT," "),target.focus_set())),
                            ("⌫",4,_bs),("CLR",5,lambda:target.delete(0,tk.END))]:
            b=tk.Button(sp,text=lbl,width=wd,height=1,font=SETUP_KBD_SP,bg=C["kbd_key"],fg=C["subtext"],
                         relief=tk.FLAT,bd=0,cursor="hand2",command=cmd)
            b.pack(side=tk.LEFT,padx=3)
            b.bind("<Enter>",lambda e,btn=b:ho(btn)); b.bind("<Leave>",lambda e,btn=b:hf(btn))


# ── Depth Setup Dialog ────────────────────────────────────────────────────────
class DepthSetupDialog(SetupDialog):
    def __init__(self, parent):
        self.method = "bm"
        super().__init__(parent, DEPTH_CONFIG_PATH, STEREO_BM_LABELS, yaml_section=None)
        self.title("SETUP — stereo_config.yaml")

    def _load(self):
        try:
            raw = yaml.safe_load(self.path.read_text()) or {}
            self.method = raw.get("stereo_method","bm")
            self.data   = dict(raw.get(f"stereo_{self.method}",{}))
            self.labels = STEREO_SGBM_LABELS if self.method=="sgbm" else STEREO_BM_LABELS
        except Exception as e: messagebox.showerror("Load Error",str(e))

    def _build(self):
        C = self._C
        hdr=tk.Frame(self,bg="#080a0f",height=56); hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr,text="⚙",fg=C["accent"],bg="#080a0f",font=F_XL_B).pack(side=tk.LEFT,padx=(20,6),pady=8)
        tk.Label(hdr,text="STEREO CONFIG",fg=C["accent"],bg="#080a0f",font=F_LG_B).pack(side=tk.LEFT,pady=8)
        b=tk.Button(hdr,text="✕  CLOSE",command=self.destroy,bg=C["red"],fg="white",
                    font=F_MD_B,relief=tk.FLAT,padx=18,pady=10,cursor="hand2",bd=0)
        b.pack(side=tk.RIGHT,padx=20,pady=10)
        b.bind("<Enter>",lambda e:b.config(bg="#cc2240")); b.bind("<Leave>",lambda e:b.config(bg=C["red"]))
        tk.Frame(self,bg=C["border_hi"],height=2).pack(fill=tk.X)

        mb=tk.Frame(self,bg="#0a0d14",height=54); mb.pack(fill=tk.X); mb.pack_propagate(False)
        tk.Label(mb,text="ALGORITHM:",fg=C["dim"],bg="#0a0d14",font=F_MD_B).pack(side=tk.LEFT,padx=(20,12),pady=14)
        self._bm=self._mbtn(mb,"BM","bm"); self._sg=self._mbtn(mb,"SGBM","sgbm")
        self._refresh_btns()
        tk.Frame(mb,bg=C["dim"],width=1).pack(side=tk.LEFT,fill=tk.Y,padx=16,pady=10)
        self._minfo=tk.Label(mb,text=self._mdesc(),fg=C["subtext"],bg="#0a0d14",font=F_SM)
        self._minfo.pack(side=tk.LEFT,pady=14)
        tk.Frame(self,bg=C["border"],height=1).pack(fill=tk.X)
        self._top=tk.Frame(self,bg=C["sidebar"]); self._top.pack(fill=tk.BOTH,expand=True)
        self._build_param_list()
        self._bot=tk.Frame(self,bg=C["editor_bg"])

    def _mbtn(self,parent,text,key):
        C=self._C
        b=tk.Button(parent,text=text,font=F_MD_B,relief=tk.FLAT,padx=22,pady=8,
                    cursor="hand2",bd=0,command=lambda m=key:self._switch(m))
        b.pack(side=tk.LEFT,padx=4,pady=10); return b

    def _refresh_btns(self):
        C=self._C
        for b,k in [(self._bm,"bm"),(self._sg,"sgbm")]:
            a=(self.method==k)
            b.config(bg=C["accent"] if a else C["kbd_key"],fg="#0d0f14" if a else C["subtext"])

    def _mdesc(self):
        return ("Semi-Global Block Matching  •  better accuracy, slower"
                if self.method=="sgbm" else "Block Matching  •  faster, suited for real-time")

    def _switch(self,m):
        self.method=m
        try:
            raw=yaml.safe_load(self.path.read_text()) or {}
            raw["stereo_method"]=m
            with open(self.path,'w') as f: yaml.dump(raw,f)
        except Exception: pass
        self._load(); self._refresh_btns(); self._minfo.config(text=self._mdesc())
        if self._edit_mode: self._exit_edit()
        for w in self._top.winfo_children(): w.destroy()
        self._cards={}; self.current_key=None; self._build_param_list()


# ══════════════════════════════════════════════════════════════════════════════
# ── Main Window
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(tk.Tk):
    _LVL = {10:"DEBUG",20:"INFO",30:"WARN",40:"ERROR",50:"FATAL"}

    def __init__(self):
        super().__init__()
        self.title("INDY ROS2 HMI")
        self.attributes('-fullscreen', FULLSCREEN)
        self.configure(bg=BG_COLOR)
        self._ros_node = self._ros_worker = self._start_pub = None
        self._build_ui()
        self._clock_timer()
        self._init_ros()
        self._set_status("READY")

    def _build_ui(self):
        sb = tk.Frame(self, bg="#0a0c11", height=54); sb.pack(fill=tk.X); sb.pack_propagate(False)
        tk.Label(sb, text="⬡",       fg=ACCENT_COLOR, bg="#0a0c11", font=F_XL_B).pack(side=tk.LEFT, padx=10)
        tk.Label(sb, text="INDY ROS2",fg=ACCENT_COLOR, bg="#0a0c11", font=F_LG_B).pack(side=tk.LEFT)
        self._st  = tk.Label(sb, text="INIT", fg=DIM_COLOR,    bg="#0a0c11", font=F_MD_B); self._st.pack(side=tk.LEFT, padx=20)
        self._clk = tk.Label(sb, text="",     fg=DIM_COLOR,    bg="#0a0c11", font=F_MD);   self._clk.pack(side=tk.RIGHT, padx=20)

        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL); pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        lf = tk.Frame(pw, bg=PANEL_COLOR); cf = tk.Frame(pw, bg=PANEL_COLOR); rf = tk.Frame(pw, bg=PANEL_COLOR)
        pw.add(lf, weight=1); pw.add(cf, weight=2); pw.add(rf, weight=3)

        # Left
        def sec(t): tk.Label(lf, text=t, fg=DIM_COLOR, bg=PANEL_COLOR, font=F_SM_B).pack(anchor='w', padx=10, pady=(10,0))
        def btn(t, c, bg, fg): tk.Button(lf, text=t, command=c, bg=bg, fg=fg, **BTN).pack(fill=tk.X, padx=10, pady=3)
        sec("CONTROLS")
        self._start_btn = tk.Button(lf, text="▶  START", command=self._on_start, bg="#1a3a1f", fg=GREEN_COLOR, **BTN)
        self._start_btn.pack(fill=tk.X, padx=10, pady=5)
        self._stop_btn  = tk.Button(lf, text="■  STOP",  command=self._on_stop,  bg="#3a1a1a", fg=RED_COLOR,   **BTN)
        self._stop_btn.pack(fill=tk.X, padx=10, pady=5)
        tk.Frame(lf, height=4, bg=PANEL_COLOR).pack()
        sec("CONFIGURATION")
        btn("⚙  SETUP",       self._open_setup, "#2a2010", YELLOW_COLOR)
        btn("⚙  DEPTH SETUP", self._open_depth, "#2a2010", YELLOW_COLOR)
        tk.Frame(lf, height=4, bg=PANEL_COLOR).pack()
        sec("LOG")
        btn("🗑  CLEAR LOGS", self._clear_logs, "#111c26", ACCENT_COLOR)
        if show_qr_window:
            btn("📱  QR CODE", lambda: show_qr_window(self), "#0e1a1a", ACCENT_COLOR)
        tk.Frame(lf, height=4, bg=PANEL_COLOR).pack()
        sec("SYSTEM STATUS")
        sf = tk.Frame(lf, bg="#0f1218"); sf.pack(fill=tk.X, padx=10, pady=5)
        self._dot_ros = tk.Label(sf, text="● ROS2 Node", fg=DIM_COLOR, bg="#0f1218", font=F_MD); self._dot_ros.pack(anchor='w')
        self._dot_pub = tk.Label(sf, text="● Publisher",  fg=DIM_COLOR, bg="#0f1218", font=F_MD); self._dot_pub.pack(anchor='w')

        # Center
        tk.Label(cf, text="CAMERAS", fg=DIM_COLOR, bg=PANEL_COLOR, font=F_SM_B).pack(anchor='w', padx=10, pady=(10,0))
        self.cam_detect = CameraWidget(cf, "DETECT IMAGE", IMAGE_TOPIC_DETECT); self.cam_detect.pack(fill=tk.BOTH, expand=True, pady=5)
        self.cam_depth  = CameraWidget(cf, "DEPTH IMAGE",  IMAGE_TOPIC_DEPTH);  self.cam_depth.pack(fill=tk.BOTH, expand=True, pady=5)

        # Right
        tk.Label(rf, text="LOGS", fg=DIM_COLOR, bg=PANEL_COLOR, font=F_SM_B).pack(anchor='w', padx=10, pady=(10,0))
        self.log_coords  = LogPanel(rf, "COORDINATES", GREEN_COLOR);  self.log_coords.pack(fill=tk.BOTH, expand=True, pady=2)
        self.log_collect = LogPanel(rf, "COLLECT",     YELLOW_COLOR); self.log_collect.pack(fill=tk.BOTH, expand=True, pady=2)

    def _clock_timer(self):
        self._clk.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._clock_timer)

    def _set_status(self, text, color=None):
        self._st.config(text=text, fg=color or DIM_COLOR)

    def _init_ros(self):
        if not HAS_ROS:
            self._dot_ros.config(fg=RED_COLOR); return
        try:
            rclpy.init(args=None)
            self._ros_node  = rclpy.create_node("hmi_node")
            self._start_pub = self._ros_node.create_publisher(StartMsg, START_TOPIC, 10)
            executor = MultiThreadedExecutor(num_threads=3)
            executor.add_node(self._ros_node)
            self._ros_worker = RosWorker(self._ros_node, executor)

            sub = self._ros_node.create_subscription
            sub(RosImage, IMAGE_TOPIC_DETECT,
                lambda m: self._ros_worker.submit_decode(IMAGE_TOPIC_DETECT, m, self.cam_detect), 1)
            sub(RosImage, IMAGE_TOPIC_DEPTH,
                lambda m: self._ros_worker.submit_decode(IMAGE_TOPIC_DEPTH,  m, self.cam_depth),  1)

            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
            sub(RosLog, "/rosout", self._on_rosout,
                QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                           durability=DurabilityPolicy.VOLATILE,
                           history=HistoryPolicy.KEEP_LAST, depth=100))

            threading.Thread(target=self._ros_worker.spin, daemon=True).start()
            self._dot_ros.config(fg=GREEN_COLOR)
            self._dot_pub.config(fg=GREEN_COLOR)
        except Exception as e:
            self._dot_ros.config(fg=RED_COLOR)
            print(f"[HMI] ROS init error: {e}")

    def _on_rosout(self, msg):
        lvl  = self._LVL.get(msg.level, f"L{msg.level}")
        line = f"[{lvl}] [{msg.name}] {msg.msg}"
        nl   = msg.name.lower().lstrip('/')
        ml   = msg.msg.lower()
        ch   = 'coords' if COORD_KW.lower() in ml else 'collect' if COLLECT_KW.lower() in nl else None
        if ch:  # only dispatch to coord/collect; skip everything else
            self.after(0, lambda l=line, c=ch: self._dispatch(l, c))

    def _dispatch(self, line, ch):
        if ch == 'coords':   self.log_coords.append(line)
        elif ch == 'collect': self.log_collect.append(line)

    def _pub_start(self, v):
        if not HAS_ROS or not self._start_pub: return
        try:
            m = StartMsg(); m.start = v; self._start_pub.publish(m)
        except Exception: pass

    def _on_start(self): self._pub_start(True);  self._set_status("STARTED", GREEN_COLOR)
    def _on_stop(self):  self._pub_start(False); self._set_status("STOPPED", RED_COLOR)
    def _clear_logs(self): self.log_coords.clear(); self.log_collect.clear()
    def _open_setup(self): SetupDialog(self, SETUP_PATH, SETUP_LABELS, yaml_section="setup").focus_set()
    def _open_depth(self): DepthSetupDialog(self).focus_set()

    def _shutdown(self):
        if self._ros_worker:
            try: self._ros_worker.stop()
            except: pass
        if self._ros_node:
            try: self._ros_node.destroy_node(); rclpy.shutdown()
            except: pass
        self.quit()

    def closeEvent(self): self._shutdown(); self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global app
    def _sig(s, f):
        print("\nShutting down...")
        try: app._shutdown()
        except: pass
        sys.exit(0)
    _signal.signal(_signal.SIGINT,  _sig)
    _signal.signal(_signal.SIGTERM, _sig)
    app = MainWindow()
    app.protocol("WM_DELETE_WINDOW", app.closeEvent)
    app.mainloop()

if __name__ == "__main__":
    main()
