"""
hmi_launcher/log_bus.py
─────────────────────────────────────────────────────────────────────────────
Shared log bus — cầu nối log giữa Tkinter HMI và Flask web server.

Cách dùng:
  from hmi_launcher.log_bus import bus

  # Ghi log (từ bất kỳ module nào):
  bus.push('log_main',   'some message')
  bus.push('log_coords', 'Updated target: x=1.2 y=3.4')

  # Đăng ký callback nhận log (Tkinter HMI):
  bus.subscribe(my_callback)   # callback(channel, line)

  # Lấy snapshot toàn bộ log (web server):
  all_logs = bus.snapshot()    # {'log_main': [...], ...}
─────────────────────────────────────────────────────────────────────────────
"""

import threading
from datetime import datetime

CHANNELS = ('log_main', 'log_coords', 'log_collect')
MAX_LINES = 400


class LogBus:
    def __init__(self):
        self._lock        = threading.Lock()
        self._logs        = {ch: [] for ch in CHANNELS}
        self._subscribers = []   # list of callables: fn(channel, line)

    # ── Ghi log ──────────────────────────────────────────────────────────────
    def push(self, channel: str, message: str):
        """Ghi 1 dòng log vào channel, thông báo tất cả subscriber."""
        if channel not in self._logs:
            channel = 'log_main'
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}" if not message.startswith('[') else message

        with self._lock:
            buf = self._logs[channel]
            buf.append(line)
            if len(buf) > MAX_LINES:
                del buf[0]
            subs = list(self._subscribers)

        # Gọi callback ngoài lock để tránh deadlock
        for fn in subs:
            try:
                fn(channel, line)
            except Exception:
                pass

    def clear(self, channel: str = None):
        """Xóa log — channel cụ thể hoặc tất cả nếu channel=None."""
        with self._lock:
            if channel:
                self._logs.get(channel, []).clear()
            else:
                for ch in CHANNELS:
                    self._logs[ch].clear()
        for fn in list(self._subscribers):
            try:
                fn('__clear__', channel or 'all')
            except Exception:
                pass

    # ── Đọc log ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """Trả về copy toàn bộ log hiện tại (dùng cho web server khi client connect)."""
        with self._lock:
            return {ch: list(lines) for ch, lines in self._logs.items()}

    def tail(self, channel: str, n: int = 50) -> list:
        """Lấy n dòng cuối của một channel."""
        with self._lock:
            return list(self._logs.get(channel, [])[-n:])

    # ── Subscribe / unsubscribe ───────────────────────────────────────────────
    def subscribe(self, fn):
        """Đăng ký callback fn(channel, line) — nhận mọi dòng log mới."""
        with self._lock:
            if fn not in self._subscribers:
                self._subscribers.append(fn)

    def unsubscribe(self, fn):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not fn]

    # ── Helper: phân loại log từ subprocess stdout ───────────────────────────
    def classify_and_push(self, line: str,
                          coord_kw: str = "Updated target:",
                          collect_kw: str = "collect_logger_node"):
        """
        Tự động chọn channel dựa theo nội dung dòng log.
        Dùng trong _read_proc() của cả GUI lẫn server.
        """
        low = line.lower()
        if coord_kw.lower() in low:
            self.push('log_coords', line)
        elif collect_kw.lower() in low:
            self.push('log_collect', line)
        else:
            self.push('log_main', line)


# ── Singleton dùng chung toàn app ─────────────────────────────────────────────
bus = LogBus()