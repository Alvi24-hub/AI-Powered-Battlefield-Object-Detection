import asyncio
import websockets
import cv2

async def stream_video():
    uri = "ws://localhost:8000/ws/v1/stream"
    video_path = "170300-843059179.mp4" 
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}. Make sure the video file is in this folder!")
        return

    print(f"Connecting to AegisVision stream at {uri} using {video_path}...")
    async with websockets.connect(uri) as websocket:
        print("Connected! Streaming video frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            _, encoded_img = cv2.imencode('.jpg', frame)
            image_bytes = encoded_img.tobytes()
            
            await websocket.send(image_bytes)
            response = await websocket.recv()
            print(f"Server Response: {response}")
            
            await asyncio.sleep(0.03)

if __name__ == "__main__":
    try:
        asyncio.run(stream_video())
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
