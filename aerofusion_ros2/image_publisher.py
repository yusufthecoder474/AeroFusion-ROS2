"""Synthetic image publisher; Stage-2 supports finite demo runs."""
import rclpy
import time
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from .frames import generate_rgb_frame

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.declare_parameter('width', 320)
        self.declare_parameter('height', 240)
        self.declare_parameter('fps', 2.0)
        self.declare_parameter('max_frames', 0)  # 0 = continuous (Stage 1 behavior)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.max_frames = int(self.get_parameter('max_frames').value)
        if self.width < 64 or self.height < 64 or not 0 < self.fps <= 30 or self.max_frames < 0:
            raise ValueError('width/height >=64; fps in (0,30]; max_frames >=0')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.counter = 0
        self.started = time.monotonic()
        self.finished = False
        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)
        self.get_logger().info('Publishing synthetic RGB8 frames on /camera/image_raw')

    def publish_frame(self):
        if self.finished:
            rclpy.shutdown()
            return
        if self.publisher.get_subscription_count() == 0:
            if time.monotonic() - self.started > 30:
                raise RuntimeError('No image subscriber discovered within 30 seconds')
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'synthetic_camera'
        msg.height = self.height
        msg.width = self.width
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = self.width * 3
        msg.data = generate_rgb_frame(self.width, self.height, self.counter)
        self.publisher.publish(msg)
        self.counter += 1
        if self.counter <= 3 or self.counter % 5 == 0:
            self.get_logger().info(f'Published synthetic frame {self.counter} ({self.width}x{self.height})')
        if self.max_frames and self.counter >= self.max_frames:
            self.get_logger().info(f'Published {self.max_frames} frames; closing demo')
            self.finished = True
            self.timer.cancel()
            self.timer = self.create_timer(2.0, self.publish_frame)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
