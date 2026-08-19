import asyncio
import cv2
import json
import websockets

VIDEO_PATH = "battlefield_demo.mp4"

async def stream_video(server_uri="ws://localhost:8000/ws/v1/stream"):
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"\n❌ Error: Could not open '{VIDEO_PATH}'. Please verify the file name and path.\n")
        return

    print(f"\n🚀 Connecting to Pipeline Hub at {server_uri}...\n")
    
    try:
        async with websockets.connect(server_uri) as websocket:
            print("🟢 Connected! Streaming video frames...\n")
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("🔁 End of video reached. Looping stream...\n")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # 📺 Show the actual video playing in a pop-up window!
                cv2.imshow("Battlefield Live Video Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Encode frame to JPEG
                success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not success:
                    continue

                # Send JPEG bytes over WebSocket
                await websocket.send(buffer.tobytes())
                frame_count += 1

                # Receive telemetry output
                response = await websocket.recv()
                telemetry = json.loads(response)

                lat = telemetry.get('latencies_ms', {})
                status = telemetry.get('status', 'ok')
                targets = telemetry.get('targets', [])

                print(f" Frame {frame_count:03d} | Status: {status}")
                print(f"   ├─ Stage 1 (CLAHE):       {lat.get('preprocessing', 0):.2f} ms")
                print(f"   ├─ Stage 2 (YOLO):        {lat.get('detection', 0):.2f} ms")
                print(f"   ├─ Stage 3 (Tracker):     {lat.get('tracking', 0):.2f} ms")
                print(f"   └─ Total Pipeline:        {lat.get('total_pipeline', 0):.2f} ms")

                print(f"   🎯 Active Targets Detected: {len(targets)}")
                for tgt in targets:
                    cls = tgt.get('class_name', tgt.get('class', 'target')).upper()
                    conf = tgt.get('confidence', tgt.get('conf', 1.0)) * 100
                    wp = tgt.get('weak_point', tgt.get('target_part', 'N/A'))
                    t_status = tgt.get('status', 'ACTIVE')
                    t_id = tgt.get('target_id', tgt.get('track_id', 'T-01'))
                    print(f"       • [{t_id}] {cls} ({conf:.0f}%) -> Weak Point: {wp} | Status: {t_status}")

                print("-" * 60)
                await asyncio.sleep(0.033)

    except websockets.exceptions.ConnectionClosed:
        print("\nStream connection closed cleanly.")
    except Exception as e:
        print(f"\n⚠️ Stream error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n🔒 Video stream stopped.")

if __name__ == "__main__":
    asyncio.run(stream_video())

