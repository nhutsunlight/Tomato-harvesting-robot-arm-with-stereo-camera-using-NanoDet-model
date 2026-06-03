#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference
from res_msgs.msg import PoseRes
from res_msgs.msg import ResFlag
from connect_msgs.msg import ConnectMsg
from connect_msgs.msg import ConnectStatus
from start_msgs.msg import StartMsg
from skip_signal_msgs.msg import SkipSignal
from move_signal_msgs.msg import MoveSignal
import onnxruntime as ort
from ament_index_python.packages import get_package_share_directory
import os
import time

bridge = CvBridge()

set_conf        = 0.5
set_class_score = 0.5
STRIDES         = [8, 16, 32, 64]
INFERENCE_EVERY_N_FRAMES = 2
FIRST_DETECT_STABLE_FRAMES = 5
FIRST_DETECT_IOU_THRESH = 0.5

pkg_path   = get_package_share_directory('yolobot_recognition_py')
model_path = os.path.join(pkg_path, 'models', 'nanodet_model4.onnx')
label_path = os.path.join(pkg_path, 'models', 'name.txt')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def softmax_last(x):
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x

def build_grid(num_anchors, input_h, input_w, strides=STRIDES):
    grids = []
    for stride in strides:
        fh = input_h // stride
        fw = input_w // stride
        ys, xs = np.mgrid[0:fh, 0:fw]
        s_arr  = np.full((fh * fw,), stride, dtype=np.float32)
        block  = np.stack([xs.ravel() * stride,
                           ys.ravel() * stride,
                           s_arr], axis=1)
        grids.append(block)
        if sum(len(g) for g in grids) >= num_anchors:
            break
    grid = np.concatenate(grids, axis=0)
    if len(grid) < num_anchors:
        pad  = np.tile(grid[-1:], (num_anchors - len(grid), 1))
        grid = np.concatenate([grid, pad], axis=0)
    return grid[:num_anchors].astype(np.float32)


class RecognitionNode(Node):

    def __init__(self):
        super().__init__('recognition_node')
        cv2.setUseOptimized(True)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads     = 2
        opts.inter_op_num_threads     = 1
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern       = True
        opts.enable_mem_reuse         = True

        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=['CPUExecutionProvider'])

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        inp_shape = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape

        self.INPUT_H     = inp_shape[2]
        self.INPUT_W     = inp_shape[3]
        self.NUM_ANCHORS = out_shape[1]

        with open(label_path) as f:
            self.classes = [c for c in f.read().split('\n') if c.strip()]
        self.NUM_CLASSES = len(self.classes)

        self.GRID      = build_grid(self.NUM_ANCHORS, self.INPUT_H, self.INPUT_W, STRIDES)
        self._mean     = np.array([103.53, 116.28, 123.675], np.float32)
        self._inv_std  = (1.0 / np.array([57.375, 57.12, 58.395], np.float32)).astype(np.float32)
        self._dfl_proj = None

        self._lb_canvas = np.zeros((self.INPUT_H, self.INPUT_W, 3), dtype=np.uint8)
        self._inp_buf   = np.empty((1, 3, self.INPUT_H, self.INPUT_W), dtype=np.float32)
        self._tmp_buf   = np.empty((self.INPUT_H, self.INPUT_W, 3),    dtype=np.float32)

        # Subscriptions
        self.subscription = self.create_subscription(
            Image, '/stereo/left/img_for_yolo', self.camera_callback, 10)
        self.pause_sub = self.create_subscription(
            PoseRes, '/pose_res', self.pause_callback, 10)
        self.start_sub = self.create_subscription(
            StartMsg, '/start_msg', self.start_callback, 10)
        self.skip_sub = self.create_subscription(
            SkipSignal, '/skip_signal', self.skip_callback, 10)

        # Publishers
        self.yolov8_pub   = self.create_publisher(Yolov8Inference, '/Yolov8_Inference', 10)
        self.raw_img_pub  = self.create_publisher(Image, '/stereo/left/image_raw_det', 10)  # raw cho visualize node
        self.pose_res_pub = self.create_publisher(PoseRes, '/pose_res', 10)
        self.connect_pub  = self.create_publisher(ConnectMsg, '/connect_msg', 10)
        self.move_signal_pub = self.create_publisher(MoveSignal, '/move_signal', 10)

        self.time       = 0.0
        self.start      = False
        self.frame_count = 0

        self.no_detection_start_   = None
        self.no_detection_timeout_ = 3.0
        self.first_dectect = True
        self.first_detect_stable_count_ = 0
        self.first_detect_ref_ = None

        self._img_h  = 0
        self._img_w  = 0
        self._margin = 0
        self._lb_scale = 0.0
        self._lb_top   = 0
        self._lb_left  = 0
        self._lb_new_h = 0
        self._lb_new_w = 0
        self._lb_ready = False

    @staticmethod
    def _stamp_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def pause_callback(self, msg: PoseRes):
        if msg.pose_res:
            self.time  = float(msg.pose_res[0].x)

    def start_callback(self, msg: StartMsg):
        self.start = msg.start

    def skip_callback(self, msg: SkipSignal):
        if msg.skip:
            self.first_dectect = True
            self.first_detect_stable_count_ = 0
            self.first_detect_ref_ = None
            self.get_logger().info("Skip signal received. Reset first detection.")

    def reset_request(self):
        connect_msg    = ConnectMsg()
        connect_status = ConnectStatus()
        connect_status.wait_key   = False
        connect_status.connection = False
        connect_status.harvest_flag = False
        connect_msg.connect_msg.append(connect_status)
        self.connect_pub.publish(connect_msg)

        res_msg  = PoseRes()
        res_flag = ResFlag()
        #res_flag.flag = True
        res_flag.skip = False
        res_msg.pose_res.append(res_flag)
        self.pose_res_pub.publish(res_msg)

    def move_signal_publish(self, move):
        move_msg = MoveSignal()
        move_msg.move = move
        self.move_signal_pub.publish(move_msg)
        if move is True:
            time.sleep(2)  # Ensure the message is sent before any state change

    def _init_frame_params(self, img0):
        self._img_h  = img0.shape[0]
        self._img_w  = img0.shape[1]
        self._margin = int(min(self._img_w, self._img_h) * 0.01)

        scale  = min(self.INPUT_W / self._img_w, self.INPUT_H / self._img_h)
        new_w  = int(self._img_w * scale)
        new_h  = int(self._img_h * scale)
        top    = (self.INPUT_H - new_h) // 2
        left   = (self.INPUT_W - new_w) // 2

        self._lb_scale = scale
        self._lb_top   = top
        self._lb_left  = left
        self._lb_new_h = new_h
        self._lb_new_w = new_w

        self._lb_canvas[:top, :, :]        = 0
        self._lb_canvas[top+new_h:, :, :]  = 0
        self._lb_canvas[:, :left, :]       = 0
        self._lb_canvas[:, left+new_w:, :] = 0
        self._lb_ready = True

    def _ensure_frame_params(self, img0):
        if (not self._lb_ready or
                img0.shape[0] != self._img_h or
                img0.shape[1] != self._img_w):
            self._init_frame_params(img0)

    def _letterbox_fast(self, image):
        cv2.resize(
            image, (self._lb_new_w, self._lb_new_h),
            dst=self._lb_canvas[
                self._lb_top  : self._lb_top  + self._lb_new_h,
                self._lb_left : self._lb_left + self._lb_new_w],
            interpolation=cv2.INTER_LINEAR)
        return self._lb_canvas

    def _preprocess_fast(self, img):
        np.subtract(img, self._mean, out=self._tmp_buf, casting='unsafe')
        np.multiply(self._tmp_buf, self._inv_std, out=self._tmp_buf)
        self._inp_buf[0] = self._tmp_buf.transpose(2, 0, 1)
        return self._inp_buf

    def _postprocess_fast(self, out):
        preds = out[0]
        n     = min(len(preds), len(self.GRID))
        preds = preds[:n]
        grid  = self.GRID[:n]

        cls = preds[:, :self.NUM_CLASSES]
        if cls.max() > 10:
            cls = sigmoid(cls)

        conf = cls.max(axis=1)
        cid  = cls.argmax(axis=1)

        keep = conf > set_conf
        if not keep.any():
            return None

        preds_k = preds[keep]
        conf_k  = conf[keep]
        cid_k   = cid[keep]
        grid_k  = grid[keep]
        box_raw = preds_k[:, self.NUM_CLASSES:]

        if box_raw.shape[1] > 4:
            reg_max = box_raw.shape[1] // 4 - 1
            if self._dfl_proj is None or len(self._dfl_proj) != reg_max + 1:
                self._dfl_proj = np.arange(reg_max + 1, dtype=np.float32)
            reg  = box_raw.reshape(-1, 4, reg_max + 1)
            dist = (softmax_last(reg) * self._dfl_proj).sum(axis=2)
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
                x1, y1, x2, y2 = (box_raw[:, 0], box_raw[:, 1],
                                   box_raw[:, 2], box_raw[:, 3])

        inv_scale = 1.0 / self._lb_scale
        x1 = (x1 - self._lb_left) * inv_scale
        y1 = (y1 - self._lb_top)  * inv_scale
        x2 = (x2 - self._lb_left) * inv_scale
        y2 = (y2 - self._lb_top)  * inv_scale

        x1 = np.clip(x1, 0, self._img_w - 1)
        y1 = np.clip(y1, 0, self._img_h - 1)
        x2 = np.clip(x2, 0, self._img_w - 1)
        y2 = np.clip(y2, 0, self._img_h - 1)

        ws = x2 - x1
        hs = y2 - y1
        valid = (ws >= 5) & (hs >= 5)
        if not valid.any():
            return None

        boxes_np  = np.stack([x1[valid], y1[valid], ws[valid], hs[valid]],
                             axis=1).astype(np.int32)
        scores_np = conf_k[valid]
        cids_np   = cid_k[valid].astype(np.int32)
        return boxes_np, scores_np, cids_np

    def _run_inference(self, img0):
        img_lb = self._letterbox_fast(img0)
        inp    = self._preprocess_fast(img_lb)
        out    = self.session.run([self.output_name], {self.input_name: inp})[0]

        res = self._postprocess_fast(out)
        if res is None:
            return [], False

        boxes_np, scores_np, cids_np = res

        keep_indices = []
        for cid in np.unique(cids_np):
            cls_indices = np.flatnonzero(cids_np == cid)
            if cls_indices.size == 0:
                continue
            raw_indices = cv2.dnn.NMSBoxes(
                boxes_np[cls_indices].tolist(),
                scores_np[cls_indices].tolist(),
                set_conf,
                set_class_score)
            if len(raw_indices) == 0:
                continue
            keep_indices.extend(cls_indices[raw_indices.flatten()].tolist())

        if not keep_indices:
            return [], False

        indices    = sorted(keep_indices, key=lambda i: float(scores_np[i]), reverse=True)
        boxes_list = boxes_np.tolist()
        margin     = self._margin
        n_cls      = self.NUM_CLASSES

        results   = []
        has_valid = False

        for i in indices:
            if i >= len(boxes_list):
                continue
            x1, y1, w, h = boxes_list[i]
            cid  = int(cids_np[i])
            conf = float(scores_np[i])
            if cid >= n_cls:
                continue
            is_in_pic = (
                x1 > margin and y1 > margin and
                x1 + w < self._img_w - margin and
                y1 + h < self._img_h - margin)
            if is_in_pic:
                has_valid = True
            results.append((x1, y1, w, h, cid, conf, int(i), is_in_pic))

        return results, has_valid

    def _build_yolo_msg(self, results, now_stamp):
        yolo_msg = Yolov8Inference()
        yolo_msg.header.frame_id = "inference"
        yolo_msg.header.stamp    = now_stamp

        should_publish = False
        classes = self.classes

        for (x1, y1, w, h, cid, conf, orig_i, is_in_pic) in results:
            if is_in_pic:
                should_publish = True

            inf_res            = InferenceResult()
            inf_res.top        = x1
            inf_res.left       = y1
            inf_res.bottom     = w
            inf_res.right      = h
            inf_res.id         = cid
            inf_res.class_name = classes[cid]
            inf_res.conf       = conf
            inf_res.tomato_id  = orig_i
            yolo_msg.yolov8_inference.append(inf_res)

        return yolo_msg, should_publish

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, aw, ah = box_a
        bx1, by1, bw, bh = box_b
        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, aw) * max(0, ah)
        area_b = max(0, bw) * max(0, bh)
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _first_detect_is_stable(self, results):
        valid_results = [r for r in results if r[7]]
        if not valid_results:
            self.first_detect_stable_count_ = 0
            self.first_detect_ref_ = None
            return False

        signature = sorted(
            [(r[4], (r[0], r[1], r[2], r[3])) for r in valid_results],
            key=lambda item: (item[0], item[1][0], item[1][1]))

        if self.first_detect_ref_ is None:
            self.first_detect_ref_ = signature
            self.first_detect_stable_count_ = 1
            return False

        if len(signature) != len(self.first_detect_ref_):
            self.first_detect_ref_ = signature
            self.first_detect_stable_count_ = 1
            return False

        for (cid, bbox), (ref_cid, ref_bbox) in zip(signature, self.first_detect_ref_):
            if cid != ref_cid:
                self.first_detect_ref_ = signature
                self.first_detect_stable_count_ = 1
                return False
            if self._bbox_iou(bbox, ref_bbox) < FIRST_DETECT_IOU_THRESH:
                self.first_detect_ref_ = signature
                self.first_detect_stable_count_ = 1
                return False

        self.first_detect_stable_count_ += 1
        self.first_detect_ref_ = signature

        if self.first_detect_stable_count_ < FIRST_DETECT_STABLE_FRAMES:
            return False

        self.first_dectect = False
        self.first_detect_ref_ = None
        self.get_logger().info("First detection is stable.")
        return True

    def camera_callback(self, data):
        self.frame_count += 1
        self.get_logger().debug("IMAGE RECEIVED")
        if self._stamp_sec(data.header.stamp) < self.time or not self.start:
            self.get_logger().info(f"Time: {self._stamp_sec(data.header.stamp):.2f} < {self.time:.2f}, Start: {self.start}")
            return

        img0 = bridge.imgmsg_to_cv2(data, "bgr8")

        self._ensure_frame_params(img0)

        now_stamp          = data.header.stamp
        results, has_valid = self._run_inference(img0)

        if not has_valid:
            now_mono = time.monotonic()
            if self.no_detection_start_ is None:
                self.no_detection_start_ = now_mono
            elif now_mono - self.no_detection_start_ >= self.no_detection_timeout_:
                self.reset_request()
                self.get_logger().info("No detection timeout.")
                self.no_detection_start_ = None
                self.first_dectect = True
                #add start msg for mobilerobot hera
                self.move_signal_publish(True)
            return

        self.no_detection_start_ = None

        if self.first_dectect and not self._first_detect_is_stable(results):
            #adding stop msg for mobilerobot hera
            self.move_signal_publish(False)
            self.get_logger().info(
                f"Waiting stable first detection: {self.first_detect_stable_count_}/{FIRST_DETECT_STABLE_FRAMES}")
            return

        yolo_msg, should_publish = self._build_yolo_msg(results, now_stamp)

        if should_publish:
            self.yolov8_pub.publish(yolo_msg)
            self.get_logger().info("Inference results published.")
        else:
            self.reset_request()

        # Pub raw image cho visualize node dùng (dùng lại header gốc)
        raw_msg = bridge.cv2_to_imgmsg(img0, "bgr8")
        raw_msg.header = data.header
        self.raw_img_pub.publish(raw_msg)


def main():
    rclpy.init(args=None)
    rclpy.spin(RecognitionNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
