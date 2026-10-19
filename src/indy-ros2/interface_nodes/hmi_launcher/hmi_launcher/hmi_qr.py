"""
hmi_qr.py — QR code window cho Tkinter HMI.

Thêm vào launcher_gui_3.py:
  1. import: from hmi_launcher.hmi_qr import show_qr_window
  2. Trong _build_ui(), thêm button:
       tk.Button(left, text="📱  QR CODE",
                 command=lambda: show_qr_window(self),
                 bg="#0e1a1a", fg=ACCENT_COLOR, **BUTTON_STYLE
                ).pack(fill=tk.X, padx=10, pady=3)
"""

import tkinter as tk
import threading
from PIL import Image as PILImage


def _get_server_url(port: int = 5000) -> str:
    import requests
    try:
        res = requests.get(f"http://localhost:{port}/api/qr", timeout=1)
        data = res.json()
        return data.get("url", f"http://localhost:{port}")
    except:
        # fallback nếu server chưa chạy
        return f"http://localhost:{port}"


def _make_qr_image(url: str):
    """Tạo PIL Image của QR code. Trả về None nếu thư viện chưa cài."""
    try:
        import qrcode
        from PIL import Image as PILImage
        qr = qrcode.QRCode(
            version=3,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8, border=3,
        )
        qr.add_data(url)
        qr.make(fit=True)
        return qr.make_image(fill_color="#00e5ff", back_color="#0d0f14")
    except ImportError:
        return None
    except Exception as e:
        print(f"[QR] Error: {e}")
        return None


def show_qr_window(parent, port: int = 5000):
    """
    Hiện cửa sổ QR code fullscreen (hoặc popup lớn).
    Gọi từ bất kỳ đâu trong Tkinter:
        show_qr_window(self)   # self = MainWindow hoặc bất kỳ widget nào
    """
    url = _get_server_url(port)

    win = tk.Toplevel(parent)
    win.title("QR Code — Web HMI")
    win.configure(bg="#0d0f14")
    win.attributes('-fullscreen', True)

    # Header
    header = tk.Frame(win, bg="#0a0c11", height=52)
    header.pack(fill=tk.X, side=tk.TOP)
    header.pack_propagate(False)
    tk.Label(header, text="📱  SCAN TO OPEN WEB HMI",
             fg="#00e5ff", bg="#0a0c11",
             font=("Courier New", 14, "bold")).pack(side=tk.LEFT, padx=20)
    btn_close = tk.Button(header, text="✕  CLOSE", command=win.destroy,
                          bg="#ff3b5c", fg="white",
                          font=("Courier New", 10, "bold"),
                          relief=tk.FLAT, padx=16, pady=8, cursor="hand2", bd=0)
    btn_close.pack(side=tk.RIGHT, padx=16, pady=8)

    tk.Frame(win, bg="#00e5ff", height=2).pack(fill=tk.X)

    # Body
    body = tk.Frame(win, bg="#0d0f14")
    body.pack(fill=tk.BOTH, expand=True)

    # URL label (hiện ngay)
    tk.Label(body, text=url, fg="#00e5ff", bg="#0d0f14",
             font=("Courier New", 18, "bold")).pack(pady=(40, 10))
    tk.Label(body, text="Mở trên điện thoại cùng WiFi",
             fg="#4a5568", bg="#0d0f14",
             font=("Courier New", 12)).pack()

    # QR image placeholder
    qr_lbl = tk.Label(body, bg="#0d0f14",
                      text="Đang tạo QR...", fg="#4a5568",
                      font=("Courier New", 11))
    qr_lbl.pack(pady=24)

    # Tạo QR trong thread để không block GUI
    def _load_qr():
        import requests
        try:
            res = requests.get(f"http://localhost:{port}/api/qr", timeout=2)
            data = res.json()

            url = data.get("url")
            qr_b64 = data.get("qr")

            if not qr_b64:
                raise Exception("No QR data")

            from PIL import ImageTk
            import base64, io

            img_data = base64.b64decode(qr_b64)
            img = PILImage.open(io.BytesIO(img_data))

            size = min(win.winfo_screenwidth(), win.winfo_screenheight()) - 280
            size = max(240, min(size, 480))
            img = img.resize((size, size))

            photo = ImageTk.PhotoImage(img)

            def _update():
                qr_lbl.config(image=photo, text="")
                qr_lbl.image = photo

            win.after(0, _update)

        except Exception as e:
            win.after(0, lambda e=e: qr_lbl.config(
                text=f"QR error: {e}", fg="#ff3b5c"))

    threading.Thread(target=_load_qr, daemon=True).start()

    # Hint
    tk.Label(body,
             text="Hoặc gõ địa chỉ trên vào browser điện thoại",
             fg="#4a5568", bg="#0d0f14",
             font=("Courier New", 10)).pack(pady=(8, 0))