"""ROS-independent image processing and JSON payloads used by the real ROS node."""
import time
from .detectors import detect_synthetic_red_box, validate_rgb_image
from .tracking import IoUTracker

class PerceptionPipeline:
    def __init__(self, mode='synthetic', detector=None, min_iou=.3, max_misses=2,
                 reset_each_frame=False):
        if mode not in ('synthetic','onnx'): raise ValueError('mode must be synthetic or onnx')
        if mode=='onnx' and detector is None: raise ValueError('ONNX mode requires a real detector')
        self.mode=mode;self.detector=detector;self.min_iou=min_iou;self.max_misses=max_misses
        self.reset_each_frame=reset_each_frame
        self.tracker=IoUTracker(min_iou,max_misses);self.received=0
        self.last_frame_id=None;self.last_stamp=None;self.last_dimensions=None

    def process(self,width,height,step,data,encoding,frame_id,stamp_ns):
        started=time.perf_counter()
        if encoding not in ('rgb8','bgr8'):raise ValueError('Expected rgb8 or bgr8')
        validate_rgb_image(width,height,step,data)
        if not isinstance(stamp_ns,int) or stamp_ns<0:raise ValueError('Invalid timestamp')
        if self.mode=='synthetic':
            if encoding!='rgb8':raise ValueError('Synthetic detector requires rgb8')
            detections=detect_synthetic_red_box(width,height,step,data)
        else:detections=self.detector.detect(width,height,step,data,encoding)
        # Never associate across camera changes, time rewinds, or unrelated DET stills.
        reset=(self.reset_each_frame or self.last_frame_id!=frame_id or self.last_dimensions!=(width,height) or
               (self.last_stamp is not None and stamp_ns<=self.last_stamp))
        if reset:self.tracker=IoUTracker(self.min_iou,self.max_misses)
        tracks=self.tracker.update(detections);self.received+=1
        self.last_frame_id=frame_id;self.last_stamp=stamp_ns;self.last_dimensions=(width,height)
        common=dict(schema_version=1,source='red-square-rule' if self.mode=='synthetic' else 'aerodetect-onnx-yolov8n',
                    frame_id=frame_id,stamp_ns=stamp_ns,image_number=self.received,width=width,height=height,
                    association_reset=reset,tracking_scope='independent_image' if self.reset_each_frame else 'ordered_sequence',
                    processing_ms=round((time.perf_counter()-started)*1000,3))
        boxes=[dict(label=d.label,confidence=d.confidence,bbox_xyxy=list(d.bbox)) for d in detections]
        return dict(common,objects=boxes),dict(common,detections=len(detections),tracks=tracks)
