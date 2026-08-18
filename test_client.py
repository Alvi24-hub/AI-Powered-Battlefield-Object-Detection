import asyncio
import cv2
import numpy as np
import websockets
import json

async def stream_mock_frames():
    uri = "ws://localhost:8000/ws/v1/stream"
    print(f"Connecting to Pipeline Hub at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(" Connected! Streaming test frames...\n")
            
            for i in range(1, 11):
                mock_frame = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', mock_frame)
                frame_bytes = buffer.tobytes()
                
                await websocket.send(frame_bytes)
                
                response = await websocket.recv()
                telemetry = json.loads(response)
                
                print(f" Frame {i:02d} | Status: {telemetry['status']} | Frame Shape: {telemetry['frame_shape']}")
                print(f"   ⏱️  Stage 1 (Alvira CLAHE):    {telemetry['latencies_ms']['preprocessing']} ms")
                print(f"   ⏱️  Stage 2 (Vishwadeep YOLO):  {telemetry['latencies_ms']['detection']} ms")
                print(f"   ⏱️  Stage 3 (Pushpa Tracker):  {telemetry['latencies_ms']['tracking']} ms")
                print(f"   ⏱️  Total Pipeline Latency:   {telemetry['latencies_ms']['total_pipeline']} ms")
                
                targets = telemetry.get("targets", [])
                print(f"   🎯 Active Tracked Targets ({len(targets)}):")
                for tgt in targets:
                    print(f"      • [{tgt['target_id']}] {tgt['class'].upper()} ({tgt['confidence']*100:.0f}%) -> Weak Point: {tgt['weak_point']} | Status: {tgt['status']}")
                
                print("-" * 65)
                await asyncio.sleep(0.1)

    except websockets.exceptions.ConnectionClosed:
        print("\n Stream connection closed cleanly.")
    except Exception as e:
        print(f"\n Stream interrupted: {e}")

if __name__ == "__main__":
    asyncio.run(stream_mock_frames())
