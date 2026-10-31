"""ROS 2 Image -> detection -> IoU tracking -> JSON String topic.

Output is intentionally simple JSON; no claim of vision_msgs compatibility.
"""
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .pipeline import PerceptionPipeline
from .detectors import AeroDetectONNX


class TrackingNode(Node):
    def __init__(self, **kwargs):
        super().__init__('aerofusion_tracker', **kwargs)
        self.declare_parameter('mode', 'synthetic')
        self.declare_parameter('min_iou', 0.3)
        self.declare_parameter('max_misses', 2)
        self.declare_parameter('aerodetect_root', '')
        self.declare_parameter('onnx_path', '')
        self.declare_parameter('model_size', 640)
        self.declare_parameter('threads', 1)
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('reset_each_frame', False)

        self.mode = str(self.get_parameter('mode').value)
        self.onnx = None
        if self.mode == 'onnx':
            self.onnx = AeroDetectONNX(
                self.get_parameter('aerodetect_root').value,
                self.get_parameter('onnx_path').value,
                int(self.get_parameter('model_size').value),
                int(self.get_parameter('threads').value),
                float(self.get_parameter('confidence').value),
            )
        elif self.mode != 'synthetic':
            raise ValueError('mode must be synthetic or onnx')

        self.pipeline = PerceptionPipeline(self.mode, self.onnx,
            float(self.get_parameter('min_iou').value), int(self.get_parameter('max_misses').value),
            bool(self.get_parameter('reset_each_frame').value))
        self.detection_publisher = self.create_publisher(String, '/detection/objects', 10)
        self.rejected = 0
        self.publisher = self.create_publisher(String, '/tracking/objects', 10)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.on_image, 10)
        self.received = 0
        self.get_logger().info(f'Stage 2 started: mode={self.mode}; output=/tracking/objects')

    def on_image(self, msg):
        try:
            detection_result, result = self.pipeline.process(
                msg.width, msg.height, msg.step, msg.data, msg.encoding, msg.header.frame_id,
                msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec)
            self.received = self.pipeline.received
            tracks = result['tracks']
            detections = detection_result['objects']
            detection_msg = String()
            detection_msg.data = json.dumps(detection_result, separators=(',', ':'), allow_nan=False)
            self.detection_publisher.publish(detection_msg)
            output = String()
            output.data = json.dumps(result, separators=(',', ':'), allow_nan=False)
            self.publisher.publish(output)
            if self.received <= 3 or self.received % 5 == 0:
                ids = [track['track_id'] for track in tracks]
                self.get_logger().info(
                    f'Frame {self.received}: detections={len(detections)}, track IDs={ids}; '
                    f'{result["processing_ms"]:.1f}ms'
                )
        except (ValueError, RuntimeError, OSError) as exc:
            self.rejected += 1
            self.get_logger().error(f'Frame rejected: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrackingNode()
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
