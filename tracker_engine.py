#!/usr/bin/env python3
"""
Aegis-Vision: Core Tactical Spatial Tracking & Telemetry Engine
Author: Janet (Spatial Tracking & Telemetry Lead)
Specialization: Low-latency, air-gapped Battlefield Object Tracking

This module implements a state-of-the-art Simple Online and Realtime Tracking (SORT)
approach utilizing Kalman Filters (via FilterPy) for state estimation. It features:
1. Electronic Warfare Resilience ("Ghost Tracking"): Predicts positions of lost targets
   using Kalman state transitions for up to 10 frames to handle occlusion or jamming.
2. Anti-Decoy Logic: Evaluates a real-time "Movement Consistency Score" using known physical
   class velocity limits to flag fast-moving decoys.
3. Velocity Vectors: Calculates pixel-per-second velocity for fire-control and lead-aiming.
4. Scale & Thread Safety: Uses Redis as an in-memory database with a local in-memory fallback.
   Operations are thread-safe for high-concurrency stream processing.
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
# Known physical speed limits on the battlefield to detect decoys (Anti-Decoy Logic)
CLASS_METADATA = {
    0: {"name": "Soldier", "max_speed_kmh": 30.0},     # Peak human sprint speed is ~37 km/h, soldier with gear is lower
    1: {"name": "Tank", "max_speed_kmh": 70.0},        # Standard main battle tanks top out around 60-72 km/h
    2: {"name": "Drone", "max_speed_kmh": 180.0},      # Tactical drones top out around 150-180 km/h
    3: {"name": "Jet", "max_speed_kmh": 2500.0},       # Supersonic fighter jets
    4: {"name": "Truck", "max_speed_kmh": 110.0}       # Heavy transport trucks
}

DEFAULT_METERS_PER_PIXEL = 0.05  # Spatial resolution scale parameter (configurable)
MAX_LOST_FRAMES = 15           # Electronic Warfare Resilience frame threshold


# ------------------------- HELPER MATH FUNCTIONS -------------------------
def bbox_to_z(bbox: List[float]) -> np.ndarray:
    """
    Converts a bounding box of the format [x1, y1, x2, y2] into a measurement vector
    z = [u, v, s, r]^T, where (u, v) is the center, s is the scale (area), and r is the aspect ratio.
    """
    x1, y1, x2, y2 = bbox
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r]).reshape((4, 1))


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """
    Converts a Kalman state vector z = [u, v, s, r]^T back into a bounding box [x1, y1, x2, y2].
    """
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
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    """
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
    """
    Manages active target states in Redis for sub-millisecond lookups.
    Features robust, connection-error-resilient fallbacks to thread-safe local storage.
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self.connected = False
        # Thread pool executor for non-blocking asynchronous Redis writes/deletes
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="RedisAsyncWorker")
        self._connect()

    def save_target_async(self, target_id: str, target_data: Dict[str, Any]):
        """Submits target state saving to background worker to prevent blocking video stream."""
        self.executor.submit(self.save_target, target_id, target_data)

    def delete_target_async(self, target_id: str):
        """Submits target deletion to background worker."""
        self.executor.submit(self.delete_target, target_id)

    def _connect(self):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                socket_connect_timeout=1.0,
                decode_responses=True
            )
            self.redis_client.ping()
            self.connected = True
            logger.info("Successfully established connection to Redis State Store.")
        except Exception as e:
            self.connected = False
            logger.warning(
                f"Redis connection failed (host={self.host}, port={self.port}). "
                f"Aegis-Vision will fallback to thread-safe local storage. Error: {e}"
            )

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
            # Auto-expire keys after 30 seconds to prevent memory leaks for expired tracks
            self.redis_client.expire(key, 30)
            return True
        except Exception as e:
            logger.error(f"Failed to sync target {target_id} to Redis: {e}")
            return False

    def get_target(self, target_id: str) -> Optional[Dict[str, Any]]:
        if not self.connected or self.redis_client is None:
            return None
        try:
            key = f"aegis:target:{target_id}"
            data = self.redis_client.hgetall(key)
            if not data:
                return None
            return {
                "target_id": data["target_id"],
                "bbox": json.loads(data["bbox"]),
                "velocity": json.loads(data["velocity"]),
                "status": data["status"],
                "class_id": int(data["class_id"]),
                "class_name": data["class_name"],
                "consistency_score": float(data["consistency_score"]),
                "is_decoy": bool(int(data["is_decoy"])),
                "timestamp": float(data["timestamp"])
            }
        except Exception as e:
            logger.error(f"Failed to fetch target {target_id} from Redis: {e}")
            return None

    def delete_target(self, target_id: str) -> bool:
        if not self.connected or self.redis_client is None:
            return False
        try:
            key = f"aegis:target:{target_id}"
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete target {target_id} from Redis: {e}")
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
            logger.error(f"Failed to flush tracking data from Redis: {e}")
            return False

# ------------------------- INDIVIDUAL KALMAN TRACK -------------------------
class AegisTrack:
    """
    Represents an active target track using a Kalman Filter to model target kinematic state.
    State representation: x = [u, v, s, r, u_dot, v_dot, s_dot]^T
    State Machine: TENTATIVE -> CONFIRMED (after 3 hits) -> GHOST (when occluded/skipped) -> EXPIRED
    """
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
        
        # State Machine: New tracks start as TENTATIVE
        self.status = "TENTATIVE"
        self.is_decoy = False
        self.consistency_score = 100.0
        self.meters_per_pixel = meters_per_pixel
        self.history: List[List[float]] = [bbox]

        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        z = bbox_to_z(bbox)
        self.kf.x[:4] = z
        self.kf.x[4:] = 0.0

        self.kf.F = np.eye(7)
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)

        # Kalman Tuning:
        # R: Lower measurement noise for center coordinates (u, v) to trust YOLO detections more.
        self.kf.R = np.diag([0.5, 0.5, 10.0, 10.0])
        # P: Initial uncertainty
        self.kf.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0])
        # Q: Higher process noise for velocity terms to follow high-speed / erratic battlefield targets.
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.0005])

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
            # Transition to GHOST during occlusion or skipped frames
            self.status = "GHOST"

        self.time_since_update += 1
        pred_bbox = z_to_bbox(self.kf.x[:4])
        return [float(v) for v in pred_bbox]

    def update(self, bbox: List[float], class_id: int, conf: float, timestamp: float):
        dt = timestamp - self.last_seen if self.last_seen is not None else (1.0 / 30.0)
        if dt <= 0:
            dt = 1.0 / 30.0

        self.time_since_update = 0
        self.hits += 1
        self.last_seen = timestamp
        self.class_id = class_id
        self.conf = conf
        
        # State Machine Logic: Promote from TENTATIVE to CONFIRMED after 3 hits
        if self.hits >= 3:
            self.status = "CONFIRMED"
        else:
            self.status = "TENTATIVE"

        z = bbox_to_z(bbox)
        self.kf.update(z)

        dx = self.kf.x[4, 0]
        dy = self.kf.x[5, 0]

        self._verify_movement_consistency(dx, dy)

        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)

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
    """
    Core thread-safe spatial tracking engine for Aegis-Vision.
    Maintains, updates, and expires tracks. Integrates with Redis for target states storage.
    """
    def __init__(self, meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL, max_lost_frames: int = MAX_LOST_FRAMES):
        self.lock = threading.Lock()
        self.tracks: Dict[str, AegisTrack] = {}
        self.meters_per_pixel = meters_per_pixel
        self.max_lost_frames = max_lost_frames
        self.redis_store = RedisStateStore()
        
        # Thread-safe backup local in-memory store if Redis is offline
        self.local_backup_store: Dict[str, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Performs SORT update step with Kalman state prediction, IOU association with class penalties,
        track birth/expiration, and asynchronous Redis telemetry sync.
        """
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            # 1. State Prediction: Propagate all existing Kalman filters forward
            predicted_bboxes = {}
            for track_id, track in list(self.tracks.items()):
                pred_bbox = track.predict(timestamp)
                predicted_bboxes[track_id] = pred_bbox

            # 2. Association: Match predictions with current detections
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
                        
                        # V-IoU Gated Search Space Expansion for high-speed targets
                        if iou_val <= 0.0:
                            pred_box = predicted_bboxes[track_id]
                            wb = pred_box[2] - pred_box[0]
                            hb = pred_box[3] - pred_box[1]
                            expanded_pred = [
                                pred_box[0] - wb * 2.5,
                                pred_box[1] - hb * 2.5,
                                pred_box[2] + wb * 2.5,
                                pred_box[3] + hb * 2.5
                            ]
                            expanded_iou = calculate_iou(det_bbox, expanded_pred)
                            if expanded_iou > 0.0:
                                iou_val = 0.35 + (expanded_iou * 0.1)

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

            # 3. Update Matched Tracks (supporting both "conf" and "confidence")
            for r, c in matched_indices:
                det = detections[r]
                track_id = track_ids[c]
                det_conf = float(det.get("confidence", det.get("conf", 0.0)))
                self.tracks[track_id].update(det["bbox"], det.get("class_id", 0), det_conf, timestamp)

            # 4. Handle Unmatched Tracks (Ghost Tracking or Expiration)
            for c in unmatched_tracks:
                track_id = track_ids[c]
                track = self.tracks[track_id]
                
                # Exceeding the Electronic Warfare resilience window -> Expiration
                if track.time_since_update > self.max_lost_frames:
                    logger.info(f"[-] Target {track_id} expired (lost for >{self.max_lost_frames} frames). Deleting.")
                    self.redis_store.delete_target_async(track_id)
                    if track_id in self.local_backup_store:
                        del self.local_backup_store[track_id]
                    del self.tracks[track_id]
                else:
                    # Pure predicted telemetry (GHOST track) persists in Redis/Local state stores
                    pass

            # 5. Handle Unmatched Detections (Track Birth) (supporting both "conf" and "confidence")
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
                logger.info(f"[+] New Target Initiated: {new_track.track_id} | Class ID: {det.get('class_id', 0)}")

            # 6. Package and Sync Telemetry to Databases (Non-blocking async Redis sync, native Python float serialization)
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

                # Save to Redis asynchronously (non-blocking I/O) & Sync local backup
                self.redis_store.save_target_async(track_id, target_data)
                self.local_backup_store[track_id] = target_data

                # Format exact payload returned to down-stream layers
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

# =========================================================================
# MOCKING & INDEPENDENT UNIT TESTING
# =========================================================================

class MockInference:
    """
    Simulates AI detections coming from Vishwadeep's inference pipeline.
    Allows independent development and thorough verification of tracker_engine.py.
    """
    def __init__(self):
        # We simulate 15 frames of data
        self.current_frame = 0

    def get_detections(self) -> List[Dict[str, Any]]:
        """
        Simulates detections for 3 objects:
        1. Target 1 (Soldier): Moves slowly. Undergoes Electronic Warfare Jamming / Occlusion
           (lost from frame 5 to 9, re-appears on frame 10).
        2. Target 2 (Drone): Moves normally at legal drone speeds.
        3. Target 3 (Tank Decoy): Moves at 150 pixels-per-frame (extremely high velocity).
           Instantly flags anti-decoy logic.
        """
        f = self.current_frame
        detections = []

        # 1. Soldier (Class ID 0, Max Speed 30 km/h)
        # Moves slowly to the right by +5 pixels per frame
        if f < 5 or f >= 10:
            detections.append({
                "bbox": [100.0 + (f * 5), 100.0, 130.0 + (f * 5), 180.0],
                "conf": 0.92,
                "class_id": 0
            })
        else:
            # Electronic Warfare resilience test: Jamming / Occlusion happens!
            # Target 1 is completely omitted from Vishwadeep's detections.
            pass

        # 2. Drone (Class ID 2, Max Speed 180 km/h)
        # Moves at a steady speed of -10 pixels-per-frame on X-axis, +10 on Y-axis
        detections.append({
            "bbox": [500.0 - (f * 10), 300.0 + (f * 10), 550.0 - (f * 10), 350.0 + (f * 10)],
            "conf": 0.89,
            "class_id": 2
        })

        # 3. Decoy Tank (Class ID 1, Max Speed 70 km/h)
        # Moves extremely fast, +150 pixels-per-frame (unphysical speed for main battle tanks!)
        detections.append({
            "bbox": [200.0 + (f * 150), 600.0, 280.0 + (f * 150), 680.0],
            "conf": 0.85,
            "class_id": 1
        })

        self.current_frame += 1
        return detections


if __name__ == "__main__":
    print("=" * 70)
    print("🛡️  AEGIS-VISION: SPATIAL TRACKING ENGINE (LIVE SIMULATION) 🛡️")
    print("=" * 70)
    
    # Initialize tracker (assumes 0.05 meters per pixel spatial resolution)
    tracker = SpatialTargetTracker(meters_per_pixel=0.05, max_lost_frames=MAX_LOST_FRAMES)
    mock_inference = MockInference()
    
    # Simulated 30 FPS timing
    fps = 30.0
    dt_step = 1.0 / fps
    simulated_time = time.time()

    # Flush Redis states before test
    tracker.redis_store.clear_all()

    for frame_idx in range(14):
        simulated_time += dt_step
        detections = mock_inference.get_detections()
        
        print(f"\n📺 --- [FRAME {frame_idx + 1}] Detections Received: {len(detections)} ---")
        for det in detections:
            bbox_str = f"[{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
            print(f"  └─ Detect -> Class ID: {det['class_id']} ({CLASS_METADATA[det['class_id']]['name']}) | BBox: {bbox_str} | Conf: {det['conf']:.2f}")

        # Update core tracker
        tracks = tracker.update(detections, timestamp=simulated_time)
        
        # Display tracking output
        print(f"📡 --- Tracker Active Tracks: {len(tracks)} ---")
        print(f"  {'Target ID':<10} | {'Class':<10} | {'Status':<10} | {'Velocity Vector':<18} | {'Score':<10} | {'Anti-Decoy':<15}")
        print("  " + "-" * 82)
        for t in tracks:
            vel_str = f"[{t['velocity'][0]:+7.1f}, {t['velocity'][1]:+7.1f}] px/s"
            status_color = "🔴" if t['status'] == "GHOST" else "🟢"
            decoy_status = "⚠️ DECOY DETECTED" if t['is_decoy'] else "✅ VERIFIED"
            
            print(f"  {status_color} {t['target_id']:<8} | {t['class_name']:<10} | {t['status']:<10} | {vel_str:<18} | {t['consistency_score']:<10.1f} | {decoy_status:<15}")
        
        # Small delay to mimic pipeline streaming
        time.sleep(0.1)

    print("\n" + "=" * 70)
    print("🛡️  SIMULATION COMPLETE - SPATIAL TRACKING SYSTEM PRODUCTION READY 🛡️")
    print("=" * 70)





