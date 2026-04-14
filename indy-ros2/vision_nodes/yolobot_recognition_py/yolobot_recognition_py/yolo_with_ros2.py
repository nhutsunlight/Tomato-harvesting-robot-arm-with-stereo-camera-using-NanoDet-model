#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference
from collect_msgs.msg import CollectMsg
from collect_msgs.msg import CollectTime
from res_msgs.msg import PoseRes
from res_msgs.msg import ResFlag
from connect_msgs.msg import ConnectMsg
from connect_msgs.msg import ConnectStatus
from start_msgs.msg import StartMsg
import onnxruntime as ort
from ament_index_python.packages import get_package_share_directory
import os

bridge = CvBridge()

# model setting
set_conf        = 0.5
set_class_score = 0.4

# NanoDet grid config
STRIDES = [8, 16, 32, 64]

# ── Skip-frame: chỉ inference 1 trong N frame ──────────────────────────────
INFERENCE_EVERY_N_FRAMES = 2   # tăng lên 3 nếu Pi vẫn còn nặng

pkg_path = get_package_share_directory('yolobot_recognition_py')
model_path = os.path.join(pkg_path, 'models', 'nanodet_model4.onnx')
label_path = os.path.join(pkg_path, 'models', 'name.txt')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax_last(x):
    """Softmax dọc theo axis cuối – tối ưu hơn bản gốc."""
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def build_grid(num_anchors, input_h, input_w, strides=STRIDES):
    grids = []
    for stride in strides:
        fh = input_h // stride
        fw = input_w // stride
        ys, xs = np.mgrid[0:fh, 0:fw]          # vectorized
        s_arr  = np.full((fh * fw,), stride, dtype=np.float32)
        block  = np.stack([xs.ravel() * stride,
                           ys.ravel() * stride,
                           s_arr], axis=1)
        grids.append(block)
        if sum(len(g) for g in grids) >= num_anchors:
            break

    grid = np.concatenate(grids, axis=0)

    if len(grid) < num_anchors:
        pad   = np.tile(grid[-1:], (num_anchors - len(grid), 1))
        grid  = np.concatenate([grid, pad], axis=0)

    return grid[:num_anchors].astype(np.float32)


class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        cv2.setUseOptimized(True)

        # ── ONNX session – tối ưu cho CPU nhỏ ─────────────────────────────
        opts = ort.SessionOptions()
        opts.intra_op_num_threads       = 2          # RPi 4 có 4 core; để 2 tránh starve OS
        opts.inter_op_num_threads       = 1
        opts.execution_mode             = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level   = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern         = True
        opts.enable_mem_reuse           = True

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        inp_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape

        self.INPUT_H     = inp_shape[2]
        self.INPUT_W     = inp_shape[3]
        self.NUM_ANCHORS = out_shape[1]
        self.OUT_DIM     = out_shape[2]

        with open(
            label_path
        ) as f:
            self.classes = [c for c in f.read().split('\n') if c.strip()]

        self.NUM_CLASSES = len(self.classes)

        self.GRID = build_grid(self.NUM_ANCHORS, self.INPUT_H, self.INPUT_W, STRIDES)

        # ── Precompute normalization ────────────────────────────────────────
        self._mean = np.array([103.53, 116.28, 123.675], np.float32)
        self._std  = np.array([57.375,  57.12,  58.395],  np.float32)
        # 1/std để nhân thay vì chia (nhanh hơn)
        self._inv_std = (1.0 / self._std).astype(np.float32)

        # ── Pre-allocate input buffer – tránh malloc mỗi frame ─────────────
        self._inp_buf = np.empty((1, 3, self.INPUT_H, self.INPUT_W), dtype=np.float32)

        # ── DFL projection vector (cache) ──────────────────────────────────
        # sẽ được khởi tạo lần đầu khi biết reg_max
        self._dfl_proj = None

        # ── IO binding – zero-copy output ──────────────────────────────────
        self._io_binding = self.session.io_binding()
        # Output binding (reuse buffer across frames)
        self._out_buf = None

        total_from_strides = sum(
            (self.INPUT_H // s) * (self.INPUT_W // s) for s in STRIDES
        )
        self.get_logger().info(f"Input         : {inp_shape}")
        self.get_logger().info(f"Output        : {out_shape}")
        self.get_logger().info(f"NUM_ANCHORS   : {self.NUM_ANCHORS}")
        self.get_logger().info(f"GRID shape    : {self.GRID.shape}")
        self.get_logger().info(f"NUM_CLASSES   : {self.NUM_CLASSES}")
        self.get_logger().info(f"Classes       : {self.classes}")

        if total_from_strides != self.NUM_ANCHORS:
            self.get_logger().warn(
                f"⚠ Grid mismatch: stride sum={total_from_strides} vs model={self.NUM_ANCHORS}."
            )

        self.yolov8_inference = Yolov8Inference()

        # ── Subscriptions / Publishers (không đổi) ─────────────────────────
        self.subscription = self.create_subscription(
            Image, '/stereo/left/img_for_yolo', self.camera_callback, 10)

        self.pause_sub = self.create_subscription(
            PoseRes, '/pose_res', self.pause_callback, 10)
        
        self.start_sub = self.create_subscription(
            StartMsg, '/start_msg', self.start_callback, 10)

        self.yolov8_pub   = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 10)
        self.img_pub      = self.create_publisher(Image, "/stereo/left/image_yolo", 10)
        self.pose_res_pub = self.create_publisher(PoseRes, "/pose_res", 10)
        self.connect_pub  = self.create_publisher(ConnectMsg, "/connect_msg", 10)

        self.pause      = False
        self.time       = 0
        self.skip_frame = False
        self.start      = False

        self.no_detection_start_   = None
        self.no_detection_timeout_ = 30.0
        self.frame_count           = 0

        # ── Cache kết quả frame trước để reuse khi skip ────────────────────
        self._last_img_msg  = None
        self._last_inference = None

    # ── pause_callback / reset_request: giữ nguyên ─────────────────────────
    def pause_callback(self, msg: PoseRes):
        if msg.pose_res:
            self.pause_flag      = msg.pose_res[0].pause
            self.time       = msg.pose_res[0].x
            self.skip_frame = True
            if not self.pause_flag:
                self.pause = False
            self.get_logger().info(f"Pause: {self.pause}")

    def start_callback(self, msg: StartMsg):
        self.start = msg.start
        self.get_logger().info("Received start signal")

    def reset_request(self):
        connect_msg    = ConnectMsg()
        connect_status = ConnectStatus()
        connect_status.wait_key   = True
        connect_status.connection = False
        connect_msg.connect_msg.append(connect_status)
        self.connect_pub.publish(connect_msg)

        res_msg  = PoseRes()
        res_flag = ResFlag()
        res_flag.flag = True
        res_msg.pose_res.append(res_flag)
        self.pose_res_pub.publish(res_msg)

    # ── Letterbox: giữ nguyên logic, nhưng dùng INTER_LINEAR (nhanh nhất) ──
    def letterbox_image(self, image, target_h, target_w):
        ih, iw = image.shape[:2]
        scale  = min(target_w / iw, target_h / ih)
        new_w  = int(iw * scale)
        new_h  = int(ih * scale)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top    = (target_h - new_h) // 2
        left   = (target_w - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return canvas, scale, top, left

    # ── Preprocess: dùng buffer pre-allocated, tránh alloc mới ─────────────
    def preprocess(self, img):
        # normalize in-place vào buffer
        tmp = (img.astype(np.float32) - self._mean) * self._inv_std   # [H,W,3]
        self._inp_buf[0] = tmp.transpose(2, 0, 1)                      # [1,3,H,W]
        return self._inp_buf

    # ── Postprocess: vectorized hoàn toàn, không loop Python cho decode ────
    def postprocess(self, out, img_w, img_h, scale, top_pad, left_pad):
        preds = out[0]
        n     = min(len(preds), len(self.GRID))
        preds = preds[:n]
        grid  = self.GRID[:n]

        cls  = preds[:, :self.NUM_CLASSES]
        if cls.max() > 10:
            sigmoid(cls)           # in-place không được trực tiếp, nhưng…
            cls = sigmoid(cls)     # OK – array nhỏ

        conf = cls.max(axis=1)
        cid  = cls.argmax(axis=1)

        keep_idx = np.where(conf > set_conf)[0]
        if len(keep_idx) == 0:
            return [], [], []

        preds_k = preds[keep_idx]
        conf_k  = conf[keep_idx]
        cid_k   = cid[keep_idx]
        grid_k  = grid[keep_idx]

        box_raw = preds_k[:, self.NUM_CLASSES:]

        if box_raw.shape[1] > 4:
            reg_max = box_raw.shape[1] // 4 - 1

            # Cache DFL projection vector
            if self._dfl_proj is None or len(self._dfl_proj) != reg_max + 1:
                self._dfl_proj = np.arange(reg_max + 1, dtype=np.float32)

            reg  = box_raw.reshape(-1, 4, reg_max + 1)
            dist = (softmax_last(reg) * self._dfl_proj).sum(axis=2)   # [N,4] vectorized

            cx, cy, stride = grid_k[:, 0], grid_k[:, 1], grid_k[:, 2]
            x1 = cx - dist[:, 0] * stride
            y1 = cy - dist[:, 1] * stride
            x2 = cx + dist[:, 2] * stride
            y2 = cy + dist[:, 3] * stride
        else:
            if box_raw.max() <= 1.5:
                x1 = box_raw[:, 0] * self.INPUT_W
                y1 = box_raw[:, 1] * self.INPUT_H
                x2 = box_raw[:, 2] * self.INPUT_W
                y2 = box_raw[:, 3] * self.INPUT_H
            else:
                x1, y1, x2, y2 = box_raw[:, 0], box_raw[:, 1], box_raw[:, 2], box_raw[:, 3]

        # Remove letterbox – vectorized
        x1 = (x1 - left_pad) / scale
        y1 = (y1 - top_pad)  / scale
        x2 = (x2 - left_pad) / scale
        y2 = (y2 - top_pad)  / scale

        ws = x2 - x1
        hs = y2 - y1

        # Lọc box quá nhỏ – vectorized
        valid = (ws >= 5) & (hs >= 5)
        x1, y1, ws, hs = x1[valid], y1[valid], ws[valid], hs[valid]
        conf_k = conf_k[valid]
        cid_k  = cid_k[valid]

        boxes  = np.stack([x1, y1, ws, hs], axis=1).astype(np.int32).tolist()
        scores = conf_k.tolist()
        cids   = cid_k.tolist()

        return boxes, scores, cids

    def camera_callback(self, data):
        self.frame_count += 1

        img_timestamp = data.header.stamp
        now = self.get_clock().now().to_msg()
        self.yolov8_inference.yolov8_inference.clear()

        if self.pause or img_timestamp.sec < self.time or self.start == False:
            self.get_logger().info(f"Pause Check 111111111111111111111111111111111111")
            return

        # ── Skip-frame: decode ảnh nhưng bỏ qua inference ─────────────────
        should_infer = (self.frame_count % INFERENCE_EVERY_N_FRAMES == 0)

        img0   = bridge.imgmsg_to_cv2(data, "bgr8")
        img_h, img_w = img0.shape[:2]
        margin = int(min(img_w, img_h) * 0.01)

        if not should_infer and self._last_inference is not None:
            # Reuse kết quả frame trước, chỉ vẽ lại + publish ảnh
            boxes, confidences, classes_ids, indices = self._last_inference
        else:
            img_lb, scale, top_pad, left_pad = self.letterbox_image(
                img0, self.INPUT_H, self.INPUT_W)
            inp = self.preprocess(img_lb)

            out = self.session.run([self.output_name], {self.input_name: inp})[0]

            boxes, confidences, classes_ids = self.postprocess(
                out, img_w, img_h, scale, top_pad, left_pad)

            self.get_logger().info(f"[Frame {self.frame_count}] {len(boxes)} detections before NMS")

            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, set_conf, set_class_score) if boxes else []
            if len(indices) > 0:
                indices = indices.flatten()

            self._last_inference = (boxes, confidences, classes_ids, indices)

        # ── No-detection timeout ────────────────────────────────────────────
        if len(indices) == 0:
            if self.no_detection_start_ is None:
                self.no_detection_start_ = self.get_clock().now().seconds_nanoseconds()[0]
            else:
                elapsed = (self.get_clock().now().seconds_nanoseconds()[0]
                           - self.no_detection_start_)
                if elapsed >= self.no_detection_timeout_:
                    self.reset_request()
                    self.get_logger().info("No detection timeout.")
                    self.no_detection_start_ = None
            return

        self.no_detection_start_ = None
        should_publish = False
        count = 0

        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp    = now

        COLOR = {0: (255, 0, 0), 1: (0, 17, 255), 2: (0, 255, 26), 3: (0, 165, 255)}

        for i in indices:
            if i >= len(boxes) or i >= len(classes_ids):
                continue

            x1, y1, w, h = boxes[i]
            cid  = classes_ids[i]
            conf = confidences[i]

            if cid >= len(self.classes):
                continue

            label = self.classes[cid]
            is_in_pic = (x1 > margin and y1 > margin and
                         x1 + w < img_w - margin and
                         y1 + h < img_h - margin)

            if is_in_pic:
                should_publish  = True
                self.skip_frame = False

            # ── Populate message (không đổi) ───────────────────────────────
            inf_res             = InferenceResult()
            inf_res.top         = int(x1)
            inf_res.left        = int(y1)
            inf_res.bottom      = int(w)
            inf_res.right       = int(h)
            inf_res.id          = cid
            inf_res.class_name  = label
            inf_res.conf        = float(conf)
            inf_res.tomato_id   = int(i)
            self.yolov8_inference.yolov8_inference.append(inf_res)

            # ── Vẽ bounding box ────────────────────────────────────────────
            color = COLOR.get(cid, (255, 255, 255))
            text  = f"{label} {conf:.2f}" if cid != 1 else f"{label} {i} {conf:.2f}"
            if cid == 1:
                count += 1
            cv2.rectangle(img0, (x1, y1), (x1+w, y1+h), color, 10)
            cv2.putText(img0, text, (x1, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 5)

        self.get_logger().info(f"Riped tomatoes: {count}")

        img_msg = bridge.cv2_to_imgmsg(img0, "bgr8")
        self.img_pub.publish(img_msg)

        if should_publish:
            self.get_logger().info(
                f"Published {len(self.yolov8_inference.yolov8_inference)} detections.")
            self.yolov8_pub.publish(self.yolov8_inference)
            self.yolov8_inference.yolov8_inference.clear()
            self.pause = True
        else:
            self.reset_request()


def main():
    rclpy.init(args=None)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()