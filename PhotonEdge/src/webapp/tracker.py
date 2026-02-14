# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.

"""Multi-object Kalman tracker for PhotonEdge perception stack.

Provides persistent object ID tracking across frames using
Kalman filtering for state estimation and Hungarian assignment
for detection-to-track matching.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanTrack:
    """Single object state estimator using a constant-velocity Kalman filter.

    State vector: [x, y, w, h, vx, vy]  (bbox center-x, center-y, width, height, velocities)
    """

    _next_id = 0

    def __init__(self, x: float, y: float, w: float, h: float):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1

        self.state = np.array([x + w / 2, y + h / 2, w, h, 0, 0], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5]).astype(np.float32)
        self.R = np.eye(4, dtype=np.float32) * 10.0

        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[:4, :4] = np.eye(4)

        self.age = 0
        self.hits = 1
        self.misses = 0
        self.label = ""
        self.confidence = 0.0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self.bbox

    def update(self, bbox):
        x, y, w, h = bbox
        z = np.array([x + w / 2, y + h / 2, w, h], dtype=np.float32)
        y_res = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.P = (np.eye(6, dtype=np.float32) - K @ self.H) @ self.P
        self.hits += 1
        self.misses = 0

    @property
    def bbox(self):
        cx, cy, w, h = self.state[:4]
        return [cx - w / 2, cy - h / 2, w, h]

    @property
    def velocity(self):
        return self.state[4:6].tolist()


class MultiObjectTracker:
    """Track multiple objects across frames with persistent IDs.

    Uses IoU-based cost matrix + Hungarian assignment for matching,
    and Kalman filters for state prediction/smoothing.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.25):
        self.tracks: list = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    @staticmethod
    def _iou(box1, box2) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-6)

    def update(self, detections: list) -> list:
        """Match detections to existing tracks and return active tracks.

        Args:
            detections: list of dicts with 'bbox' [x,y,w,h], 'label', 'confidence'.

        Returns:
            List of tracked objects with stable IDs, velocities, and ages.
        """
        for track in self.tracks:
            track.predict()

        if not detections and not self.tracks:
            return []

        if not detections:
            for track in self.tracks:
                track.misses += 1
            self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
            return self._active()

        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return self._active()

        n_t = len(self.tracks)
        n_d = len(detections)
        cost = np.ones((n_t, n_d), dtype=np.float32)
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - self._iou(trk.bbox, det['bbox'])

        row_idx, col_idx = linear_sum_assignment(cost)

        matched_t, matched_d = set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < (1.0 - self.iou_threshold):
                self.tracks[r].update(detections[c]['bbox'])
                self.tracks[r].label = detections[c].get('label', '')
                self.tracks[r].confidence = detections[c].get('confidence', 0)
                matched_t.add(r)
                matched_d.add(c)

        for i in range(n_t):
            if i not in matched_t:
                self.tracks[i].misses += 1

        for j in range(n_d):
            if j not in matched_d:
                self._create_track(detections[j])

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return self._active()

    def _create_track(self, det: dict):
        x, y, w, h = det['bbox']
        t = KalmanTrack(x, y, w, h)
        t.label = det.get('label', '')
        t.confidence = det.get('confidence', 0)
        self.tracks.append(t)

    def _active(self) -> list:
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits or t.age <= self.min_hits:
                bx = [int(round(v)) for v in t.bbox]
                out.append({
                    'id': t.id,
                    'bbox': bx,
                    'label': t.label,
                    'confidence': round(t.confidence, 2),
                    'velocity': [round(v, 1) for v in t.velocity],
                    'age': t.age,
                })
        return out

    def reset(self):
        self.tracks.clear()
        KalmanTrack._next_id = 0
