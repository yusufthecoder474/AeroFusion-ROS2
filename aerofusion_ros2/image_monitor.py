import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class CameraMonitor(Node):
    def __init__(self):
        super().__init__('camera_monitor')
        self.count = 0
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.on_frame, 10)
        self.get_logger().info('Subscribed to /camera/image_raw (monitor only; no model inference)')

    def on_frame(self, msg: Image):
        expected = int(msg.width) * int(msg.height) * 3
        if msg.encoding != 'rgb8' or msg.step != msg.width * 3 or len(msg.data) != expected:
            self.get_logger().error('Malformed or unsupported image received')
            return
        self.count += 1
        if self.count <= 3 or self.count % 10 == 0:
            self.get_logger().info(
                f'Received frame {self.count}: {msg.width}x{msg.height}, '
                f'frame_id={msg.header.frame_id}, bytes={len(msg.data)}'
            )

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraMonitor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
