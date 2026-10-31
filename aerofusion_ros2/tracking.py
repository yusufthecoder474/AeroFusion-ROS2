"""Deterministic IoU tracking for small recorded-frame demonstrations.

This is not a re-identification model, optical flow tracker, ByteTrack, or MOT benchmark.
No ROS/ONNX dependency so it can be unit tested without robotics hardware.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 (image pixels)
    label: str
    confidence: float

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        if not all(isfinite(float(x)) for x in self.bbox):
            raise ValueError('Bounding box must contain finite values')
        if not x1 < x2 or not y1 < y2:
            raise ValueError('Bounding box must have positive area')
        if not self.label:
            raise ValueError('Detection label cannot be empty')
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('Detection confidence must be in [0,1]')


@dataclass
class _Track:
    track_id: int
    detection: Detection
    missed: int = 0
    age: int = 1


def intersection_over_union(a, b):
    """Intersection / union for x1,y1,x2,y2, safe for disjoint boxes."""
    overlap_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_w * overlap_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    """Greedy class-aware association. IDs persist during short, overlapping motion."""

    def __init__(self, iou_threshold=0.3, max_misses=2):
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError('IoU threshold must be in (0,1]')
        if type(max_misses) is not int or max_misses < 0:
            raise ValueError('max_misses must be a non-negative integer')
        self.iou_threshold = float(iou_threshold)
        self.max_misses = max_misses
        self._next_id = 1
        self._tracks = {}

    def update(self, detections):
        """Return visible tracked detections as list[dict], ordered by ID.

        Inactive tracks can survive max_misses frames; they are not published
        until observed again. No ID guarantee across occlusion/scene changes.
        """
        detections = list(detections)
        if not all(isinstance(d, Detection) for d in detections):
            raise TypeError('Each item must be a Detection')

        candidates = []
        for track_id, state in self._tracks.items():
            for i, detection in enumerate(detections):
                if detection.label != state.detection.label:
                    continue
                score = intersection_over_union(state.detection.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    candidates.append((-score, track_id, i))
        candidates.sort()  # high IoU first, deterministic tie-breaking

        used_tracks, used_detections = set(), set()
        visible_ids = set()
        for _, track_id, detection_idx in candidates:
            if track_id in used_tracks or detection_idx in used_detections:
                continue
            state = self._tracks[track_id]
            state.detection = detections[detection_idx]
            state.missed = 0
            state.age += 1
            used_tracks.add(track_id)
            used_detections.add(detection_idx)
            visible_ids.add(track_id)

        for track_id, state in list(self._tracks.items()):
            if track_id not in used_tracks:
                state.missed += 1
                if state.missed > self.max_misses:
                    del self._tracks[track_id]

        for i, detection in enumerate(detections):
            if i not in used_detections:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id=track_id, detection=detection)
                visible_ids.add(track_id)

        return [
            {
                'track_id': tid,
                'label': self._tracks[tid].detection.label,
                'confidence': round(self._tracks[tid].detection.confidence, 6),
                'bbox_xyxy': list(self._tracks[tid].detection.bbox),
                'age_frames': self._tracks[tid].age,
            }
            for tid in sorted(visible_ids)
        ]
