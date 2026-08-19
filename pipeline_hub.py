import time
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from main import ThermalImageProcessor
from vision_engine import LocalVisionEngine
from tracker_engine import SpatialTargetTracker

app = FastAPI(title="AegisVision Pipeline Hub")

preprocessor = ThermalImageProcessor()
detector = LocalVisionEngine()
tracker = SpatialTargetTracker(meters_per_pixel=0.05, max_lost_frames=10)

frame_count = 0

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found in directory</h1>"

@app.websocket("/ws/v1/stream")
async def pipeline_stream(websocket: WebSocket):
    global frame_count
    await websocket.accept()
    
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            t_start = time.perf_counter()
            
            if hasattr(preprocessor, 'process_frame'):
                enhanced_frame, _ = preprocessor.process_frame(image_bytes)
            else:
                np_arr = np.frombuffer(image_bytes, np.uint8)
                enhanced_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if enhanced_frame is None:
                continue
                
            frame_count += 1
            t_pre_start = time.perf_counter()
            lat_preproc = round((time.perf_counter() - t_pre_start) * 1000, 2)

            t_det_start = time.perf_counter()
            if frame_count % 3 == 0:
                inf_res = detector.run_inference(enhanced_frame)
                detections = inf_res.get("detections", []) if isinstance(inf_res, dict) else (inf_res if isinstance(inf_res, list) else [])
                inference_skipped = False
            else:
                detections = []
                inference_skipped = True
            lat_detection = round((time.perf_counter() - t_det_start) * 1000, 2)

            t_track_start = time.perf_counter()
            tracked_targets = tracker.update(detections)
            lat_tracking = round((time.perf_counter() - t_track_start) * 1000, 2)

            lat_total = round((time.perf_counter() - t_start) * 1000, 2)

            payload = {
                "frame_id": frame_count,
                "pipeline_latency_ms": lat_total,
                "inference_skipped": inference_skipped,
                "stage_latencies": {
                    "preprocessor_ms": lat_preproc,
                    "detection_ms": lat_detection,
                    "tracking_ms": lat_tracking
                },
                "locked_targets_count": len(tracked_targets),
                "targets": tracked_targets
            }

            await websocket.send_json(payload)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error: {e}")
