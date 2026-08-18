import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from main import processor as preprocessor
from vision_engine import detector
from tracker_engine import tracker

app = FastAPI(title="Battlefield Vision Hub")

@app.get("/")
async def get_dashboard():
    return FileResponse("index.html")

@app.websocket("/ws/v1/stream")
async def pipeline_stream(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            t_start = time.perf_counter()

            enhanced_frame, lat_preproc = preprocessor.process_frame(image_bytes)
            if enhanced_frame is None:
                continue

            frame_count += 1
            inference_skipped = (frame_count % 3 != 0)

            if inference_skipped:
                detections = []
                lat_detection = 0.0
            else:
                detection_result = detector.run_inference(enhanced_frame)
                detections = detection_result.get("detections", [])
                lat_detection = detection_result.get("inference_time_ms", 0.0)

            t_track_start = time.perf_counter()
            tracked_targets = tracker.update(detections)
            lat_tracking = round((time.perf_counter() - t_track_start) * 1000, 2)

            total_latency = round((time.perf_counter() - t_start) * 1000, 2)

            telemetry_payload = {
                "status": "success",
                "inference_skipped": inference_skipped,
                "targets": tracked_targets,
                "latencies_ms": {
                    "preprocessing": lat_preproc,
                    "detection": lat_detection,
                    "tracking": lat_tracking,
                    "total_pipeline": total_latency
                }
            }
            await websocket.send_json(telemetry_payload)
    except WebSocketDisconnect:
        pass

