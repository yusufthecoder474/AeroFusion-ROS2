"""Publish consecutive ordered image files from ONE recorded sequence, with timestamps.

Do not use unordered VisDrone-DET images to claim multi-object tracking quality.
Requires cv2 (python3-opencv or opencv-python) for image decoding.
"""
from pathlib import Path
import rclpy
import time
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

class RecordedPublisher(Node):
    def __init__(self):
        super().__init__('recorded_image_publisher')
        self.declare_parameter('image_dir', '')
        self.declare_parameter('fps', 2.0)
        self.declare_parameter('max_frames', 20)
        folder = Path(str(self.get_parameter('image_dir').value)).expanduser()
        self.fps = float(self.get_parameter('fps').value)
        self.max_frames = int(self.get_parameter('max_frames').value)
        if not 0 < self.fps <= 20 or self.max_frames <= 0:
            raise ValueError('fps must be in (0,20], max_frames must be positive')
        if not folder.is_dir():
            raise FileNotFoundError(f'Recorded image directory not found: {folder}')
        self.images = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
        if not self.images:
            raise ValueError(f'No png/jpeg files found in {folder}')
        import cv2
        self.cv2 = cv2
        self.count = 0
        self.started = time.monotonic()
        self.finished = False
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0/self.fps, self.emit_image)
        self.get_logger().info(f'Publishing {min(len(self.images),self.max_frames)} images from {folder}')

    def emit_image(self):
        if self.finished:
            rclpy.shutdown()
            return
        if self.publisher.get_subscription_count() == 0:
            if time.monotonic() - self.started > 30:
                raise RuntimeError('No image subscriber discovered within 30 seconds')
            return
        image = self.cv2.imread(str(self.images[self.count]), self.cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error(f'Cannot decode: {self.images[self.count]}')
        else:
            rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            h,w = rgb.shape[:2]
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'recorded_camera'
            msg.height, msg.width = h, w
            msg.encoding = 'rgb8'
            msg.is_bigendian = 0
            msg.step = w * 3
            msg.data = rgb.tobytes()
            self.publisher.publish(msg)
            if self.count < 3 or self.count % 5 == 0:
                self.get_logger().info(f'Published image {self.count+1}: {self.images[self.count].name}')
        self.count += 1
        if self.count >= min(self.max_frames, len(self.images)):
            self.get_logger().info('Finished recorded image replay')
            self.finished = True
            self.timer.cancel()
            self.timer = self.create_timer(2.0, self.emit_image)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RecordedPublisher()
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
