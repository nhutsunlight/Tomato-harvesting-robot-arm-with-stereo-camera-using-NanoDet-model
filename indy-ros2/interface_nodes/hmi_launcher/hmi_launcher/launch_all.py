"""
hmi_launcher/launcher_all.py

Khởi chạy đồng thời Tkinter HMI + Flask web server trong 1 process.

Design đơn giản nhất có thể:
  - rclpy.init() chỉ gọi 1 lần (monkey-patch)
  - Web server chạy trên daemon thread → tự tắt khi GUI đóng
  - SIGINT/SIGTERM → gọi thẳng app._shutdown() rồi os._exit(0)
    (giống hệt cách chạy standalone, không có wrapper phức tạp)
"""

import sys
import os
import signal
import threading


# ── Patch rclpy.init() thành no-op sau lần đầu ───────────────────────────────
def _patch_rclpy_init():
    try:
        import rclpy
        _orig = rclpy.init
        _done = [False]
        def _safe(*a, **kw):
            if _done[0]:
                return
            _done[0] = True
            _orig(*a, **kw)
        rclpy.init = _safe
    except ImportError:
        pass


def main():
    _patch_rclpy_init()

    # Init rclpy 1 lần
    try:
        import rclpy
        rclpy.init()
        print("[launcher_all] rclpy initialized")
    except ImportError:
        print("[launcher_all] rclpy not found — ROS disabled")
    except Exception as e:
        print(f"[launcher_all] rclpy.init error: {e}")

    # Import sau khi đã patch
    try:
        from hmi_launcher.hmi_server     import main as server_main
        from hmi_launcher.launcher_gui_3 import MainWindow
    except ImportError as e:
        print(f"[launcher_all] Import error: {e}")
        sys.exit(1)

    # Web server trên daemon thread — tự tắt khi main thread kết thúc
    t = threading.Thread(target=server_main, daemon=True, name="hmi-server")
    t.start()
    print("[launcher_all] Web server started")

    # Tạo GUI trên main thread
    print("[launcher_all] Starting Tkinter GUI...")
    app = MainWindow()
    app.protocol("WM_DELETE_WINDOW", app.closeEvent)

    # ── Signal handler ────────────────────────────────────────────────────────
    # Chạy GIỐNG HỆT standalone: gọi _shutdown() (kill ROS + kill proc tree)
    # rồi os._exit(0). Không dùng after(), không thread phụ.
    #
    # Tkinter trên Linux block mainloop nên SIGINT có thể không vào ngay.
    # Workaround: after() poll 100ms để Python interpreter xử lý signal.
    def _sighandler(sig, frame):
        print(f"\n[launcher_all] Signal {sig} — shutting down...")
        try:
            app._shutdown()   # kill ROS node + toàn bộ proc tree (psutil)
        except Exception as e:
            print(f"[launcher_all] shutdown error: {e}")
        try:
            app.destroy()
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    # Poll để Python xử lý signal ngay cả khi Tkinter đang block mainloop
    def _poll():
        app.after(100, _poll)
    app.after(100, _poll)

    app.mainloop()

    # mainloop thoát bình thường (user đóng cửa sổ → closeEvent đã gọi)
    print("[launcher_all] GUI closed.")
    os._exit(0)


if __name__ == "__main__":
    main()