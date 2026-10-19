import rclpy
from rclpy.node import Node
from res_msgs.msg import PoseRes, ResFlag
from connect_msgs.msg import ConnectMsg, ConnectStatus

class DualPublisher(Node):
    def __init__(self):
        super().__init__('dual_publisher')

        self.pose_pub = self.create_publisher(PoseRes, '/pose_res', 10)
        self.conn_pub = self.create_publisher(ConnectMsg, '/connect_msg', 10)
        self.conn_sub = self.create_subscription(ConnectMsg, '/connect_msg', self.conn_callback, 10)

        self.executed = False
        self.get_logger().info("🔁 Waiting for /connect_msg...")

    def conn_callback(self, msg):
        if self.executed:
            return

        if not msg.connect_msg:
            self.get_logger().warn("⚠️ Received empty /connect_msg")
            return

        status = msg.connect_msg[0]
        self.get_logger().info(f"📩 Got /connect_msg: wait_key={status.wait_key}")

        if status.wait_key:
            now_sec = self.get_clock().now().nanoseconds / 1e9

            # Publish /pose_res
            pose_msg = PoseRes()
            pose = ResFlag()
            pose.x = now_sec
            pose.flag = False
            pose_msg.pose_res.append(pose)
            self.pose_pub.publish(pose_msg)

            # Publish /connect_msg
            conn_msg = ConnectMsg()
            conn = ConnectStatus()
            conn.connection = True
            conn.wait_key = False
            conn.id = 0
            conn_msg.connect_msg.append(conn)
            self.conn_pub.publish(conn_msg)

            self.get_logger().info("✅ Published pose_res and connect_msg")
            self.executed = True

            # Shutdown sau 1 giây để đảm bảo message gửi xong
            self.create_timer(1.0, lambda: rclpy.shutdown())

def main():
    rclpy.init()
    node = DualPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
