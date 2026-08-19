#!/usr/bin/env python3
"""
Aegis-Vision: Core Tactical Spatial Tracking & Telemetry Engine
Author: Janet (Spatial Tracking & Telemetry Lead)
Specialization: Low-latency, air-gapped Battlefield Object Tracking
"""

import cv2
import numpy as np
import time
import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# ------------------------- LOGGING CONFIGURATION -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Janet] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AegisVisionTracker")

# ------------------------- GLOBAL CONFIGURATIONS -------------------------
CLASS_METADATA = {
    0: {"name": "Soldier", "max_speed_kmh": 30.0},
    1: {"name": "Tank", "max_speed_kmh": 70.0},
    2: {"name": "Drone", "max_speed_kmh": 180.0},
    3: {"name": "Jet", "max_speed_kmh": 2500.0},
    4: {"name": "Truck", "max_speed_kmh": 110.0}
}

DEFAULT_METERS_PER_PIXEL = 0.05
MAX_LOST_FRAMES = 10


# ------------------------- HELPER MATH FUNCTIONS -------------------------
def bbox_to_z(bbox: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r]).reshape((4, 1))


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    u, v, s, r = z[0, 0], z[1, 0], z[2, 0], z[3, 0]
    s = max(1e-6, s)
    r = max(1e-6, r)
    w = np.sqrt(s * r)
    h = s / w
    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2])


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = float(box_a_area + box_b_area - inter_area)
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


# ------------------------- REDIS STATE STORE -------------------------
class RedisStateStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self.connected = False
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="RedisAsyncWorker")
        self._connect()

    def save_target_async(self, target_id: str, target_data: Dict[str, Any]):
        self.executor.submit(self.save_target, target_id, target_data)

    def delete_target_async(self, target_id: str):
        self.executor.submit(self.delete_target, target_id)

    def _connect(self):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.host, port=self.port, db=self.db, socket_connect_timeout=1.0, decode_responses=True
            )
            self.redis_client.ping()
            self.connected = True
            logger.info("Successfully established connection to Redis State Store.")
        except Exception as e:
            self.connected = False
            logger.warning(f"Redis offline, falling back to local memory: {e}")

    def save_target(self, target_id: str, target_data: Dict[str, Any]) -> bool:
        if not self.connected or self.redis_client is None:
            return False
        try:
            key = f"aegis:target:{target_id}"
            serialized_data = {
                "target_id": target_data["target_id"],
                "bbox": json.dumps(target_data["bbox"]),
                "velocity": json.dumps(target_data["velocity"]),
                "status": target_data["status"],
                "class_id": target_data["class_id"],
                "class_name": target_data["class_name"],
                "consistency_score": target_data["consistency_score"],
                "is_decoy": 1 if target_data["is_decoy"] else 0,
                "timestamp": target_data["timestamp"]
            }
            self.redis_client.hset(key, mapping=serialized_data)
            self.redis_client.expire(key, 30)
            return True
        except Exception as e:
            logger.error(f"Failed to sync target {target_id} to Redis: {e}")
            return False

    def delete_target(self, target_id: str) -> bool:
        if not self.connected or self.redis_client is None:
            return False
        try:
            key = f"aegis:target:{target_id}"
            self.redis_client.delete(key)
            return True
        except Exception as e:
            return False

    def clear_all(self) -> bool:
        if not self.connected or self.redis_client is None:
            return False
        try:
            keys = self.redis_client.keys("aegis:target:*")
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            return False


# ------------------------- INDIVIDUAL KALMAN TRACK -------------------------
class AegisTrack:
    track_count = 0

    def __init__(self, bbox: List[float], class_id: int, conf: float, start_time: float, meters_per_pixel: float):
        AegisTrack.track_count += 1
        self.track_id = f"T-{100 + AegisTrack.track_count}"
        self.class_id = class_id
        self.conf = conf
        self.created_at = start_time
        self.last_seen = start_time
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.status = "LOCKED"
        self.is_decoy = False
        self.consistency_score = 100.0
        self.meters_per_pixel = meters_per_pixel

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        z = bbox_to_z(bbox)
        self.kf.x[:4] = z
        self.kf.x[4:] = 0.0

        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.kf.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0])
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

    def predict(self, timestamp: float) -> List[float]:
        dt = timestamp - self.last_seen if self.last_seen is not None else (1.0 / 30.0)
        if dt <= 0:
            dt = 1.0 / 30.0

        self.kf.F[0, 4] = dt
        self.kf.F[1, 5] = dt
        self.kf.F[2, 6] = dt

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hits = 0
            self.status = "GHOST"

        self.time_since_update += 1
        pred_bbox = z_to_bbox(self.kf.x[:4])
        return [float(v) for v in pred_bbox]

    def update(self, bbox: List[float], class_id: int, conf: float, timestamp: float):
        self.time_since_update = 0
        self.hits += 1
        self.last_seen = timestamp
        self.class_id = class_id
        self.conf = conf
        self.status = "LOCKED"

        z = bbox_to_z(bbox)
        self.kf.update(z)

        dx = self.kf.x[4, 0]
        dy = self.kf.x[5, 0]
        self._verify_movement_consistency(dx, dy)

    def _verify_movement_consistency(self, dx: float, dy: float):
        pixel_speed = np.sqrt(dx**2 + dy**2)
        speed_mps = pixel_speed * self.meters_per_pixel
        speed_kmh = speed_mps * 3.6

        meta = CLASS_METADATA.get(self.class_id, {"name": "Unknown", "max_speed_kmh": 120.0})
        max_speed = max(1.0, float(meta.get("max_speed_kmh", 120.0)))

        if speed_kmh <= max_speed:
            self.consistency_score = 100.0
            self.is_decoy = False
        else:
            overspeed_ratio = (speed_kmh - max_speed) / max_speed
            self.consistency_score = max(0.0, 100.0 - (overspeed_ratio * 100.0))
            if self.consistency_score < 75.0 and self.age > 3:
                self.is_decoy = True
            else:
                self.is_decoy = False


# ------------------------- SPATIAL TARGET TRACKER -------------------------
class SpatialTargetTracker:
    def __init__(self, meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL, max_lost_frames: int = MAX_LOST_FRAMES):
        self.lock = threading.Lock()
        self.tracks: Dict[str, AegisTrack] = {}
        self.meters_per_pixel = meters_per_pixel
        self.max_lost_frames = max_lost_frames
        self.redis_store = RedisStateStore()
        self.local_backup_store: Dict[str, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            predicted_bboxes = {}
            for track_id, track in list(self.tracks.items()):
                pred_bbox = track.predict(timestamp)
                predicted_bboxes[track_id] = pred_bbox

            matched_indices: List[Tuple[int, int]] = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(predicted_bboxes)))
            track_ids = list(predicted_bboxes.keys())

            if len(detections) > 0 and len(predicted_bboxes) > 0:
                cost_matrix = np.zeros((len(detections), len(predicted_bboxes)), dtype=np.float32)
                for d_idx, det in enumerate(detections):
                    det_bbox = det["bbox"]
                    det_class = det.get("class_id", 0)
                    for t_idx, track_id in enumerate(track_ids):
                        track = self.tracks[track_id]
                        iou_val = calculate_iou(det_bbox, predicted_bboxes[track_id])
                        class_penalty = 0.0 if det_class == track.class_id else 0.5
                        cost_matrix[d_idx, t_idx] = 1.0 - iou_val + class_penalty

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 0.95:
                        matched_indices.append((r, c))
                        if r in unmatched_detections:
                            unmatched_detections.remove(r)
                        if c in unmatched_tracks:
                            unmatched_tracks.remove(c)

            for r, c in matched_indices:
                det = detections[r]
                track_id = track_ids[c]
                det_conf = float(det.get("confidence", det.get("conf", 0.0)))
                self.tracks[track_id].update(det["bbox"], det.get("class_id", 0), det_conf, timestamp)

            for c in unmatched_tracks:
                track_id = track_ids[c]
                track = self.tracks[track_id]
                if track.time_since_update > self.max_lost_frames:
                    self.redis_store.delete_target_async(track_id)
                    if track_id in self.local_backup_store:
                        del self.local_backup_store[track_id]
                    del self.tracks[track_id]

            for r in unmatched_detections:
                det = detections[r]
                det_conf = float(det.get("confidence", det.get("conf", 0.0)))
                new_track = AegisTrack(
                    bbox=det["bbox"],
                    class_id=det.get("class_id", 0),
                    conf=det_conf,
                    start_time=timestamp,
                    meters_per_pixel=self.meters_per_pixel
                )
                self.tracks[new_track.track_id] = new_track

            output_list = []
            for track_id, track in self.tracks.items():
                curr_bbox = z_to_bbox(track.kf.x[:4])
                curr_bbox_native = [float(round(float(v), 2)) for v in curr_bbox]
                dx = float(round(float(track.kf.x[4, 0]), 2))
                dy = float(round(float(track.kf.x[5, 0]), 2))
                class_name = CLASS_METADATA.get(track.class_id, {"name": "Unknown"})["name"]

                target_data = {
                    "target_id": str(track_id),
                    "bbox": curr_bbox_native,
                    "velocity": [dx, dy],
                    "status": str(track.status),
                    "class_id": int(track.class_id),
                    "class_name": str(class_name),
                    "confidence": float(round(float(track.conf), 2)),
                    "consistency_score": float(round(float(track.consistency_score), 2)),
                    "is_decoy": bool(track.is_decoy),
                    "timestamp": float(timestamp)
                }

                self.redis_store.save_target_async(track_id, target_data)
                self.local_backup_store[track_id] = target_data

                output_list.append({
                    "target_id": str(track_id),
                    "bbox": curr_bbox_native,
                    "velocity": [dx, dy],
                    "status": str(track.status),
                    "class_name": str(class_name),
                    "consistency_score": target_data["consistency_score"],
                    "is_decoy": target_data["is_decoy"]
                })

            return output_list


if __name__ == "__main__":
    print("=== AEGIS-VISION: TRACKER ENGINE READY ===")
    tracker = SpatialTargetTracker()
    print("Tracker Engine initialized cleanly!")
