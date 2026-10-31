"""SORT-style Kalman tracker + minimal IoU baseline + synthetic benchmark.

Scope: SYNTHETIC ONLY. No real video, no VisDrone, no ROS in this file.
Uses built-in Hungarian assignment and CLEAR-MOT-style scores (not motmetrics).
Interface (both trackers): update(detections) -> list[(track_id, bbox)]
  detections: list of [x1, y1, x2, y2] floats.
  bbox returned in the same [x1, y1, x2, y2] format.
  A track is returned only in a frame where it is matched or created.
  Internally it is kept for up to max_age frames after a miss (Kalman:
  predicted forward; IoU baseline: stale box), so it can be re-acquired
  without a new ID if the next detection still matches.

Run (offline, requires NumPy only):
    python tracker_kalman.py
"""

from __future__ import annotations

import numpy as np
def linear_sum_assignment(cost):
    """Rectangular minimum-cost Hungarian assignment, without SciPy (O(n^3))."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    swapped = n > m
    matrix = cost.T if swapped else cost
    N, M = matrix.shape
    u = [0.0] * (N + 1)
    v = [0.0] * (M + 1)
    p = [0] * (M + 1)
    way = [0] * (M + 1)
    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (M + 1)
        used = [False] * (M + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, M + 1):
                if not used[j]:
                    cur = float(matrix[i0 - 1, j - 1]) - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(M + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, M + 1) if p[j] != 0]
    if swapped:
        pairs = [(j, i) for i, j in pairs]
    pairs.sort()
    return np.array([x for x, _ in pairs], dtype=int), np.array([y for _, y in pairs], dtype=int)


BBox = list[float]  # [x1, y1, x2, y2]


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _to_state(bbox: BBox) -> np.ndarray:
    """Box -> [cx, cy, w, h, vx, vy, vw, vh] with zero initial velocity."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    state = np.zeros(8)
    state[0], state[1] = cx, cy
    state[2], state[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return state


def _to_bbox(state: np.ndarray) -> BBox:
    cx, cy, w, h = state[0], state[1], state[2], state[3]
    return [float(cx - w / 2.0), float(cy - h / 2.0),
            float(cx + w / 2.0), float(cy + h / 2.0)]


class KalmanTrack:
    """One constant-velocity Kalman filter over [cx, cy, w, h, velocities]."""

    def __init__(self, track_id: int, bbox: BBox) -> None:
        self.id = track_id
        self.x = _to_state(bbox)
        # F: position += velocity (dt = 1 frame).
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        # H: observe position/size only, not velocity.
        self.H = np.zeros((4, 8))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        self.R = np.diag([4.0, 4.0, 4.0, 4.0])  # ~2 px bbox noise std.
        self.P = np.diag([10.0, 10.0, 10.0, 10.0,
                          100.0, 100.0, 100.0, 100.0])
        self.age = 0  # frames since last matched detection

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox: BBox) -> None:
        z = _to_state(bbox)[:4]
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.age = 0

    def bbox(self) -> BBox:
        return _to_bbox(self.x)


def _hungarian_match(tracks_boxes: list[BBox], detections: list[BBox],
                     iou_threshold: float = 0.3) -> tuple[list[tuple[int, int]],
                                                          list[int], list[int]]:
    """Hungarian matching on IoU cost. Returns (matches, unmatched_tracks, unmatched_dets)."""
    if not tracks_boxes or not detections:
        return [], list(range(len(tracks_boxes))), list(range(len(detections)))
    cost = np.array([[1.0 - iou(t, d) for d in detections] for t in tracks_boxes])
    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_tracks, matched_dets = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= 1.0 - iou_threshold:
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))
    unmatched_tracks = [i for i in range(len(tracks_boxes)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class KalmanTracker:
    """SORT-style tracker: Kalman predict + Hungarian matching on IoU."""

    def __init__(self, max_age: int = 10, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: list[KalmanTrack] = []
        self._next_id = 1

    def update(self, detections: list[BBox]) -> list[tuple[int, BBox]]:
        for track in self.tracks:
            track.predict()
        predicted = [t.bbox() for t in self.tracks]
        matches, unmatched_tracks, unmatched_dets = _hungarian_match(
            predicted, detections, self.iou_threshold)
        for ti, di in matches:
            self.tracks[ti].update(detections[di])
        for ti in unmatched_tracks:
            self.tracks[ti].age += 1
        for di in unmatched_dets:
            self.tracks.append(KalmanTrack(self._next_id, detections[di]))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [(t.id, t.bbox()) for t in self.tracks if t.age == 0]


class IoUTracker:
    """Minimal baseline: same matching/death rules, no motion prediction.

    The last seen box is kept unchanged during a dropout, so a moving
    object drifts away from the stale box and is more likely to be
    re-born under a new ID. That is the flaw the Kalman version fixes.
    """

    def __init__(self, max_age: int = 10, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: list[dict] = []  # {"id", "bbox", "age"}
        self._next_id = 1

    def update(self, detections: list[BBox]) -> list[tuple[int, BBox]]:
        boxes = [t["bbox"] for t in self.tracks]
        matches, unmatched_tracks, unmatched_dets = _hungarian_match(
            boxes, detections, self.iou_threshold)
        for ti, di in matches:
            self.tracks[ti]["bbox"] = list(detections[di])
            self.tracks[ti]["age"] = 0
        for ti in unmatched_tracks:
            self.tracks[ti]["age"] += 1
        for di in unmatched_dets:
            self.tracks.append({"id": self._next_id,
                                "bbox": list(detections[di]), "age": 0})
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return [(t["id"], list(t["bbox"])) for t in self.tracks if t["age"] == 0]


def generate_synthetic(seed: int = 42, n_frames: int = 200,
                       n_objects: int = 8, drop_prob: float = 0.15,
                       noise_std: float = 2.0, width: int = 640,
                       height: int = 480) -> tuple[list[list[tuple[int, BBox]]],
                                                    list[list[BBox]]]:
    """Synthetic crossing sequence. Returns (ground_truth, detections) per frame.

    Ground truth: constant-velocity boxes bouncing inside the frame, so
    paths cross. Detections: ground truth + Gaussian noise, with each
    detection independently dropped with probability drop_prob.
    """
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_objects, 2))
    vel = np.zeros((n_objects, 2))
    size = np.zeros((n_objects, 2))
    for i in range(n_objects):
        pos[i] = [60 + i * 65, 70 + (i % 4) * 85]
        # Fast enough that a stale box loses IoU after a 2-frame dropout:
        # 8-12 px/frame vs a ~35 px wide box. Kalman prediction is what
        # bridges that gap; the IoU baseline keeps the last seen box.
        vel[i] = [8.0 + 2.0 * (i % 3), (-5.0 if i % 2 else 5.0)]
        size[i] = [35 + 5 * (i % 3), 45 + 5 * (i % 2)]
    gt_frames: list[list[tuple[int, BBox]]] = []
    det_frames: list[list[BBox]] = []
    for _ in range(n_frames):
        gt: list[tuple[int, BBox]] = []
        dets: list[BBox] = []
        for i in range(n_objects):
            pos[i] += vel[i]
            for axis, limit in ((0, width), (1, height)):
                if pos[i, axis] - size[i, axis] / 2 < 0 or \
                        pos[i, axis] + size[i, axis] / 2 > limit:
                    vel[i, axis] *= -1
                    pos[i, axis] = np.clip(pos[i, axis],
                                           size[i, axis] / 2,
                                           limit - size[i, axis] / 2)
            box: BBox = [float(pos[i, 0] - size[i, 0] / 2),
                         float(pos[i, 1] - size[i, 1] / 2),
                         float(pos[i, 0] + size[i, 0] / 2),
                         float(pos[i, 1] + size[i, 1] / 2)]
            gt.append((i + 1, box))
            if rng.random() >= drop_prob:
                noisy = [float(v + rng.normal(0, noise_std)) for v in box]
                noisy[2] = max(noisy[2], noisy[0] + 10)
                noisy[3] = max(noisy[3], noisy[1] + 10)
                dets.append(noisy)
        gt_frames.append(gt)
        det_frames.append(dets)
    return gt_frames, det_frames


def evaluate(tracker, gt_frames, det_frames) -> dict:
    """Offline CLEAR-MOT-style evaluation on SYNTHETIC GT, IoU >= 0.5.

    IDF1 uses a global identity-to-identity one-to-one assignment.
    This evaluator is included to avoid motmetrics downloads; do not
    describe its results as independently validated by motmetrics.
    """
    from collections import Counter

    total_gt = total_hyp = misses = false_positives = switches = 0
    association_counts = Counter()
    last_match = {}
    for gt, dets in zip(gt_frames, det_frames):
        hyps = tracker.update(dets)
        total_gt += len(gt)
        total_hyp += len(hyps)
        if not gt or not hyps:
            misses += len(gt)
            false_positives += len(hyps)
            continue
        similarities = np.array([[iou(g[1], h[1]) for h in hyps] for g in gt])
        # Strongly penalize below-threshold matches; they are not valid matches.
        costs = np.where(similarities >= 0.5, 1.0 - similarities, 1e6)
        rows, cols = linear_sum_assignment(costs)
        valid = [(int(r), int(c)) for r, c in zip(rows, cols)
                 if similarities[r, c] >= 0.5]
        misses += len(gt) - len(valid)
        false_positives += len(hyps) - len(valid)
        for r, c in valid:
            gid, hid = gt[r][0], hyps[c][0]
            association_counts[(gid, hid)] += 1
            if gid in last_match and last_match[gid] != hid:
                switches += 1
            last_match[gid] = hid
    mota = 1.0 - (misses + false_positives + switches) / max(1, total_gt)
    gt_ids = sorted({key[0] for key in association_counts})
    hyp_ids = sorted({key[1] for key in association_counts})
    idtp = 0
    if gt_ids and hyp_ids:
        overlaps = np.array([[association_counts[(g, h)] for h in hyp_ids]
                             for g in gt_ids], dtype=float)
        rows, cols = linear_sum_assignment(-overlaps)
        idtp = int(sum(overlaps[r, c] for r, c in zip(rows, cols)))
    idf1 = 2 * idtp / (total_gt + total_hyp) if total_gt + total_hyp else 1.0
    return {"mota": mota, "idf1": idf1, "id_switches": switches}


def main() -> None:
    gt_frames, det_frames = generate_synthetic()
    results = {}
    for name, tracker in [("IoU", IoUTracker()), ("Kalman", KalmanTracker())]:
        results[name] = evaluate(tracker, gt_frames, det_frames)
    print("TRACKER,MOTA,IDF1,ID_SWITCHES")
    for name in ("IoU", "Kalman"):
        r = results[name]
        print(f"{name},{r['mota']:.4f},{r['idf1']:.4f},{r['id_switches']}")


if __name__ == "__main__":
    main()
