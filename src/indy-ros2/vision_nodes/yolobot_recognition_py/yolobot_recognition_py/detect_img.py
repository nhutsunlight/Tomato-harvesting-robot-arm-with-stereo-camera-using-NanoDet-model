#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()

COLOR = {0: (255, 0, 0), 1: (0, 17, 255), 2: (0, 255, 26), 3: (0, 165, 255)}


class VisualizeNode(Node):

    def __init__(self):
        super().__init__('visualize_node')

        # Sub raw image từ recognition_node
        self.img_sub = self.create_subscription(
            Image, '/stereo/left/image_raw_det', self.img_callback, 10)

        # Sub bbox msg
        self.bbox_sub = self.create_subscription(
            Yolov8Inference, '/Yolov8_Inference', self.bbox_callback, 10)

        self.img_pub = self.create_publisher(
            Image, '/stereo/left/image_yolo', 10)

        # Cache bbox mới nhất — img và bbox đến async nên giữ lại dùng
        self._last_bboxes: list = []
        self._last_bbox_stamp = None

    def bbox_callback(self, msg: Yolov8Inference):
        """Cache bbox list mỗi khi có msg mới."""
        self._last_bboxes = msg.yolov8_inference
        self._last_bbox_stamp = msg.header.stamp

    @staticmethod
    def _same_stamp(a, b):
        return a.sec == b.sec and a.nanosec == b.nanosec

    def img_callback(self, msg: Image):
        """Vẽ bbox lên ảnh rồi publish."""
        if (not self._last_bboxes or self._last_bbox_stamp is None or
                not self._same_stamp(self._last_bbox_stamp, msg.header.stamp)):
            # Không có bbox → pub ảnh gốc
            self.img_pub.publish(msg)
            return

        img0 = bridge.imgmsg_to_cv2(msg, "bgr8")

        for det in self._last_bboxes:
            x1   = det.top
            y1   = det.left
            w    = det.bottom
            h    = det.right
            cid  = det.id
            conf = det.conf
            label = det.class_name

            color = COLOR.get(cid, (255, 255, 255))
            text  = (f"{label} {conf:.2f}" if cid != 1
                     else f"{label} {det.tomato_id} {conf:.2f}")

            cv2.rectangle(img0, (x1, y1), (x1 + w, y1 + h), color, 10)
            cv2.putText(img0, text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 5)

        out_msg = bridge.cv2_to_imgmsg(img0, "bgr8")
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)


def main():
    rclpy.init(args=None)
    rclpy.spin(VisualizeNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
