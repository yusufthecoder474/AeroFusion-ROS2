import unittest
from aerofusion_ros2.frames import generate_rgb_frame
from aerofusion_ros2.detectors import detect_synthetic_red_box, validate_rgb_image
from aerofusion_ros2.tracking import Detection, IoUTracker, intersection_over_union


def red_detections(n, width=128, height=96):
    return detect_synthetic_red_box(width, height, width*3, generate_rgb_frame(width,height,n))


class TrackingTests(unittest.TestCase):
    def test_exact_synthetic_box(self):
        detections = red_detections(0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (8, 32, 40, 64))

    def test_movement_detected(self):
        self.assertEqual(red_detections(1)[0].bbox, (16, 32, 48, 64))

    def test_thirty_synthetic_frames_retaining_one_id(self):
        tracker = IoUTracker(0.3)
        ids = [tracker.update(red_detections(n, width=320, height=240))[0]['track_id'] for n in range(30)]
        self.assertEqual(ids, [1] * 30)

    def test_tracking_id_persists_with_overlap(self):
        tracker = IoUTracker(0.3)
        ids = [tracker.update(red_detections(n))[0]['track_id'] for n in range(5)]
        self.assertEqual(ids, [1]*5)

    def test_non_overlapping_box_creates_new_track(self):
        tracker = IoUTracker()
        first = tracker.update([Detection((0,0,10,10),'car',.9)])[0]
        second = tracker.update([Detection((100,100,110,110),'car',.9)])[0]
        self.assertNotEqual(first['track_id'],second['track_id'])

    def test_class_aware_matching(self):
        tracker = IoUTracker()
        first = tracker.update([Detection((0,0,20,20),'car',1)])[0]
        second = tracker.update([Detection((0,0,20,20),'person',1)])[0]
        self.assertNotEqual(first['track_id'],second['track_id'])

    def test_occlusion_one_frame(self):
        tracker = IoUTracker(max_misses=2)
        a=tracker.update([Detection((2,2,12,12),'car',.8)])[0]
        self.assertEqual(tracker.update([]),[])
        b=tracker.update([Detection((3,2,13,12),'car',.8)])[0]
        self.assertEqual(a['track_id'],b['track_id'])

    def test_old_track_retired(self):
        tracker = IoUTracker(max_misses=1)
        id1=tracker.update([Detection((0,0,10,10),'car',.8)])[0]['track_id']
        tracker.update([]);tracker.update([])
        id2=tracker.update([Detection((0,0,10,10),'car',.8)])[0]['track_id']
        self.assertNotEqual(id1,id2)

    def test_multiple_objects_one_to_one(self):
        tracker=IoUTracker()
        boxes=[Detection((0,0,10,10),'car',1),Detection((30,0,40,10),'car',1)]
        ids1=[o['track_id'] for o in tracker.update(boxes)]
        ids2=[o['track_id'] for o in tracker.update(boxes[::-1])]
        self.assertEqual(ids1,ids2)
        self.assertEqual(ids1,[1,2])

    def test_synthetic_rejects_empty_red(self):
        self.assertEqual(detect_synthetic_red_box(64,64,192,bytes((26,36,56))*(64*64)),[])

    def test_invalid_image_length(self):
        with self.assertRaises(ValueError):
            validate_rgb_image(64,64,192,b'bad')

    def test_noise_rejection(self):
        data=bytearray(bytes((26,36,56))*64*64)
        for x,y in [(0,0),(63,63),(20,30),(10,40),(55,2)] * 4:
            i=(y*64+x)*3;data[i:i+3]=bytes((255,0,0))
        self.assertEqual(detect_synthetic_red_box(64,64,192,data),[])

    def test_iou(self):
        self.assertAlmostEqual(intersection_over_union((0,0,10,10),(5,0,15,10)),1/3)
        self.assertEqual(intersection_over_union((0,0,10,10),(20,20,30,30)),0)

    def test_invalid_boxes_rejected(self):
        with self.assertRaises(ValueError):
            Detection((0,0,0,10),'car',.5)
        with self.assertRaises(ValueError):
            Detection((0,0,2,10),'car',1.5)

    def test_invalid_tracker_params(self):
        with self.assertRaises(ValueError): IoUTracker(0)
        with self.assertRaises(ValueError): IoUTracker(max_misses=-1)

if __name__ == '__main__':
    unittest.main()
