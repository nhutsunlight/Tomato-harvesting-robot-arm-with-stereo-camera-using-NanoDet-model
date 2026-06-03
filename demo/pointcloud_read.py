"""
disparity_to_3d.py
Dựng 3D point cloud từ file disparity .tiff đã lưu từ ROS node.
Dùng vedo để visualize (nhẹ, interactive, xoay được).

Yêu cầu:
    pip install opencv-python numpy vedo

Chạy:
    python disparity_to_3d.py                          # dùng file mặc định
    python disparity_to_3d.py disparity_raw_0.tiff     # chỉ định file khác
    python disparity_to_3d.py disparity_raw_0.tiff left.png  # kèm ảnh màu
"""

import sys
import numpy as np
import cv2
from vedo import Points, show, Axes

# ──────────────────────────────────────────────
# 1. Thông số camera từ YAML
# ──────────────────────────────────────────────

fx0 = 1590.86364   # focal length X (px) — từ projection_matrix left
fy0 = 1590.86364   # focal length Y (px)
cx0 = 954.39404    # principal point X (px)
cy0 = 539.04084    # principal point Y (px)

# baseline = Tx_px / fx = 111.43804 / 1590.86 ≈ 0.07006 m
Tx_px      = 111.43804
baseline_m = Tx_px / fx0   # = 0.07006 m ≈ 7 cm

print(f"[INFO] fx={fx0:.2f}  cx={cx0:.2f}  cy={cy0:.2f}")
print(f"[INFO] baseline = {baseline_m*100:.2f} cm")

# ──────────────────────────────────────────────
# 2. Đọc file disparity .tiff (float32)
# ──────────────────────────────────────────────

disp_file  = sys.argv[1] if len(sys.argv) > 1 else "disparity_raw_0.tiff"
color_file = sys.argv[2] if len(sys.argv) > 2 else "color_left_img.png"

disparity = cv2.imread(disp_file, cv2.IMREAD_UNCHANGED)
if disparity is None:
    raise FileNotFoundError(f"Không đọc được file: {disp_file}")

if disparity.ndim == 3:
    disparity = disparity[:, :, 0]

disparity = disparity.astype(np.float32)
h, w = disparity.shape

print(f"\n[INFO] Disparity shape : {disparity.shape}")
print(f"[INFO] Disparity range : min={disparity.min():.2f}  max={disparity.max():.2f}")

# ──────────────────────────────────────────────
# 3. Scale focal length nếu ảnh khác kích thước gốc 1920x1080
# ──────────────────────────────────────────────

W0, H0 = 1920, 1080
scale_x = w / W0
scale_y = h / H0
fx = fx0 * scale_x
fy = fy0 * scale_y
cx = cx0 * scale_x
cy = cy0 * scale_y

# ──────────────────────────────────────────────
# 4. Đọc ảnh màu (nếu có) để tô màu point cloud
# ──────────────────────────────────────────────

if color_file is not None:
    color_img = cv2.imread(color_file)
    if color_img is not None:
        color_img = cv2.resize(color_img, (w, h))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        has_color = True
        print(f"[INFO] Dùng ảnh màu: {color_file}")
    else:
        print(f"[WARN] Không đọc được ảnh màu: {color_file} → dùng colormap depth")
        has_color = False
else:
    has_color = False
    print("[INFO] Không có ảnh màu → tô màu theo depth (JET)")

# ──────────────────────────────────────────────
# 5. Valid mask: loại sentinel, zero, NaN
# ──────────────────────────────────────────────

valid_mask = (
    (disparity > 0.0) &
    (disparity < 1000.0) &
    np.isfinite(disparity)
)

print(f"[INFO] Valid pixels : {valid_mask.sum()} / {disparity.size} "
      f"({100.0 * valid_mask.sum() / disparity.size:.1f}%)")

# ──────────────────────────────────────────────
# 6. Tính XYZ trực tiếp từ công thức stereo
#    Z = fx * baseline / d
#    X = (u - cx) * Z / fx
#    Y = (v - cy) * Z / fy
# ──────────────────────────────────────────────

rows, cols = np.where(valid_mask)
d = disparity[valid_mask]

depth = fx * baseline_m / d          # Z camera (m)
X_cam = (cols - cx) * depth / fx     # X camera (sang phải)
Y_cam = (rows - cy) * depth / fy     # Y camera (xuống)
Z_cam = depth

print(f"[DEBUG] Depth range : min={depth.min():.4f}  max={depth.max():.4f} m")

# ──────────────────────────────────────────────
# 7. Lọc Z hợp lệ
# ──────────────────────────────────────────────

z_min, z_max = 0.05, 100.0
z_mask = (Z_cam > z_min) & (Z_cam < z_max)

X_cam = X_cam[z_mask]
Y_cam = Y_cam[z_mask]
Z_cam = Z_cam[z_mask]
rows_v = rows[z_mask]
cols_v = cols[z_mask]

print(f"[INFO] Points sau lọc Z [{z_min}, {z_max}]m : {len(Z_cam)}")

if len(Z_cam) == 0:
    raise RuntimeError(
        f"Không có điểm nào trong Z=[{z_min}, {z_max}]m.\n"
        f"Depth thực tế: min={depth.min():.4f}  max={depth.max():.4f} m"
    )

# ──────────────────────────────────────────────
# 8. Đổi trục khớp với C++ StereoPointCloudNode:
#    render_x =  Z_cam   (depth → trục X)
#    render_y = -X_cam   (-X cam → trục Y)
#    render_z = -Y_cam   (-Y cam → trục Z, lên trên)
# ──────────────────────────────────────────────

pts_x =  Z_cam    # depth
pts_y = -X_cam    # -X cam
pts_z = -Y_cam    # -Y cam (lên)

points = np.stack([pts_x, pts_y, pts_z], axis=1)   # (N, 3)

# ──────────────────────────────────────────────
# 9. Outlier removal (robust, như code Python kia)
# ──────────────────────────────────────────────

if points.shape[0] > 200:
    med   = np.median(points, axis=0)
    dists = np.linalg.norm(points - med, axis=1)
    md    = np.median(dists)
    mad   = np.median(np.abs(dists - md)) + 1e-6
    keep  = dists < (md + 6.0 * mad)
    points   = points[keep]
    rows_v   = rows_v[keep]
    cols_v   = cols_v[keep]
    print(f"[INFO] Sau outlier removal : {len(points)} điểm")

# ──────────────────────────────────────────────
# 9.5. SOR (Statistical Outlier Removal)
# ──────────────────────────────────────────────

def sor_filter(points, k=20, std_mul=1.0):
    if len(points) < k * 2:
        return np.ones(len(points), dtype=bool)

    # KD-tree bằng brute force (nhẹ, không cần sklearn)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)

    dists, _ = tree.query(points, k=k)   # (N, k)
    mean_dists = dists.mean(axis=1)

    global_mean = mean_dists.mean()
    global_std  = mean_dists.std()

    threshold = global_mean + std_mul * global_std
    mask = mean_dists < threshold

    return mask


# Apply SOR
try:
    sor_mask = sor_filter(points, k=20, std_mul=1.0)

    points = points[sor_mask]
    rows_v = rows_v[sor_mask]
    cols_v = cols_v[sor_mask]

    print(f"[INFO] Sau SOR filter : {len(points)} điểm")

except Exception as e:
    print(f"[WARN] SOR bị lỗi ({e}) → bỏ qua")

# ──────────────────────────────────────────────
# 10. Màu
# ──────────────────────────────────────────────

if has_color:
    colors = color_img[rows_v, cols_v]   # (N, 3) RGB uint8
else:
    # Colormap JET theo depth
    d_norm = (points[:, 0] - points[:, 0].min()) / \
             (points[:, 0].max() - points[:, 0].min() + 1e-6)
    import matplotlib.pyplot as plt
    cmap   = plt.get_cmap("jet")
    colors = (cmap(d_norm)[:, :3] * 255).astype(np.uint8)

# ──────────────────────────────────────────────
# 11. Downsample nếu quá nhiều điểm
# ──────────────────────────────────────────────

MAX_POINTS = 200_000
if len(points) > MAX_POINTS:
    idx    = np.random.choice(len(points), MAX_POINTS, replace=False)
    points = points[idx]
    colors = colors[idx]
    print(f"[INFO] Downsample còn : {MAX_POINTS} điểm")

# ──────────────────────────────────────────────
# 12. Lưu XYZ
# ──────────────────────────────────────────────

ext     = "." + disp_file.rsplit(".", 1)[-1]
out_txt = disp_file.replace(ext, "_points.txt")
np.savetxt(out_txt, points, fmt="%.4f", header="X(depth) Y(-Xcam) Z(-Ycam)")
print(f"[INFO] Đã lưu XYZ : {out_txt}")

# ──────────────────────────────────────────────
# 13. Visualize với vedo
# ──────────────────────────────────────────────

print(f"[INFO] Hiển thị {len(points)} điểm với vedo...")
print("       Chuột trái: xoay | Chuột giữa: pan | Scroll: zoom | Q: thoát")

pc = Points(points, r=2)
pc.pointcolors = colors

show(
    pc,
    Axes(pc),
    f"Point Cloud — {disp_file} ({len(points)} pts)",
    axes=1,
    viewup="z",
    interactive=True,
)