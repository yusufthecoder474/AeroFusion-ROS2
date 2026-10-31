"""Actual rclpy/DDS smoke check. Fails on missing ROS, lost frames, wrong source or IDs."""
import argparse,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--aerodetect',required=True,type=Path);p.add_argument('--model',required=True,type=Path);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
# Import the colcon-installed package; do not mask installation with source sys.path.
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from aerofusion_ros2.tracking_node import TrackingNode
from aerofusion_ros2.frames import generate_rgb_frame

rclpy.init();executor=SingleThreadedExecutor();peer=Node('aerofusion_verification_peer')
executor.add_node(peer);results=[]
try:
    for mode in ['synthetic','onnx']:
        options=[Parameter('mode',value=mode)]
        if mode=='onnx':options += [Parameter('aerodetect_root',value=str(a.aerodetect.resolve())),Parameter('onnx_path',value=str(a.model.resolve()))]
        tracker=TrackingNode(parameter_overrides=options);executor.add_node(tracker)
        detections={};tracks={}
        def collect(target):
            def receive(msg):
                data=json.loads(msg.data);target[data['stamp_ns']]=data
            return receive
        sub1=peer.create_subscription(String,'/detection/objects',collect(detections),10)
        sub2=peer.create_subscription(String,'/tracking/objects',collect(tracks),10)
        publisher=peer.create_publisher(Image,'/camera/image_raw',10)
        def wait_until(condition,seconds=15):
            deadline=time.monotonic()+seconds
            while not condition():
                if time.monotonic()>deadline:raise AssertionError(f'Timed out in {mode}; detections={len(detections)} tracks={len(tracks)}')
                executor.spin_once(timeout_sec=.05)
        wait_until(lambda:publisher.get_subscription_count()>0 and tracker.publisher.get_subscription_count()>0 and tracker.detection_publisher.get_subscription_count()>0)
        if mode=='onnx':
            import cv2
            manifest=json.loads((a.aerodetect/'config/subset.json').read_text())
            image=cv2.imread(str(a.dataset/'images'/manifest['images'][0]['filename']))
            if image is None:raise ValueError('Original VisDrone image unavailable')
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);height,width=image.shape[:2]
            count=3  # repeated real still tests persistent IDs, not video MOT quality
        else:width,height,count=320,240,30
        seen=[]
        for n in range(count):
            msg=Image();stamp=(n+1)*1000000000;msg.header.stamp.sec=n+1
            msg.header.frame_id=mode+'_verification';msg.width=width;msg.height=height
            msg.step=width*3;msg.encoding='rgb8';msg.is_bigendian=0
            msg.data=generate_rgb_frame(width,height,n) if mode=='synthetic' else image.tobytes()
            publisher.publish(msg);wait_until(lambda:stamp in detections and stamp in tracks)
            det,track=detections[stamp],tracks[stamp]
            assert det['frame_id']==track['frame_id']==msg.header.frame_id
            assert det['image_number']==track['image_number']==n+1
            assert track['detections']==len(det['objects'])==len(track['tracks'])
            assert track['source']==('red-square-rule' if mode=='synthetic' else 'aerodetect-onnx-yolov8n')
            assert all(t['age_frames']==n+1 for t in track['tracks'])
            current=[t['track_id'] for t in track['tracks']];assert current
            if seen:assert current==seen[-1]
            seen.append(current)
        # Rejected input must neither advance image_number nor publish stale predictions.
        rejected_before=tracker.rejected
        msg.data=b'bad';publisher.publish(msg);wait_until(lambda:tracker.rejected==rejected_before+1)
        assert tracker.received==count and len(detections)==len(tracks)==count
        results.append(dict(mode=mode,status='PASSED',frames=count,persistent_ids=seen[0],invalid_frame_rejected=True,
                            tracking_evidence='synthetic motion' if mode=='synthetic' else 'repeated identical VisDrone still; not real video'))
        peer.destroy_publisher(publisher);peer.destroy_subscription(sub1);peer.destroy_subscription(sub2)
        executor.remove_node(tracker);tracker.destroy_node()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='PASSED',transport='actual rclpy DDS',checks=results),indent=2));print(a.output)
finally:
    executor.shutdown();peer.destroy_node()
    if rclpy.ok():rclpy.shutdown()
