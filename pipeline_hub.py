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
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            t_start = time.perf_counter()

            enhanced_frame, lat_preproc = preprocessor.process_frame(image_bytes)
            if enhanced_frame is None:
                continue

            detections, lat_detection = detector.run_inference(enhanced_frame)
            tracked_targets, lat_tracking = tracker.update(detections)

            total_latency = round((time.perf_counter() - t_start) * 1000, 2)

            telemetry_payload = {
                "status": "success",
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
