import torch
import time
import numpy as np
from ultralytics import YOLO

class LocalVisionEngine:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLO model and places it on CUDA GPU memory if available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VISHWADEEP GPU ENGINE] Initialized on device: {self.device}")
        
        # Load model and transfer to target compute device
        self.model = YOLO(model_path).to(self.device)
        self.warmup_gpu()

    def warmup_gpu(self):
        """
        Executes a dummy tensor pass to eliminate initial CUDA kernel initialization lag.
        """
        print("[CUDA] Performing kernel warmup for zero-first-frame latency...")
        dummy_input = torch.zeros((1, 3, 640, 640), device=self.device)
        _ = self.model(dummy_input)
        print("[CUDA] GPU Kernel Warmup Complete.")

    def run_inference(self, frame_matrix: np.ndarray) -> dict:
        """
        Takes an enhanced BGR image matrix from Alvira's preprocessor, runs YOLO inference,
        and calculates precise tactical strike coordinates.
        """
        start_time = time.time()
        
        # Execute YOLO model scoring on CUDA tensor
        results = self.model(frame_matrix, verbose=False)[0]
        
        inference_latency = (time.time() - start_time) * 1000
        fps = 1000 / inference_latency if inference_latency > 0 else 60.0
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            # Calculate precise sub-centroid strike coordinate (Engine Bay / Turret Joint)
            strike_x = int(x1 + (x2 - x1) * 0.5)
            strike_y = int(y1 + (y2 - y1) * 0.3)

            detections.append({
                "class_id": cls_id,
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2],
                "strike_zone": {
                    "target_part": "Engine Compartment / Turret Joint",
                    "coordinates": [strike_x - 15, strike_y - 15, strike_x + 15, strike_y + 15]
                }
            })

        # Measure active CUDA VRAM allocation
        vram_allocated = (
            torch.cuda.memory_allocated() / (1024 ** 3) if self.device == "cuda" else 0.0
        )

        return {
            "inference_time_ms": round(inference_latency, 2),
            "fps": round(fps, 1),
            "vram_usage_gb": round(vram_allocated, 2),
            "detections": detections
        }

# ------------------------- STANDALONE UNIT TEST -------------------------
if __name__ == "__main__":
    print("Running standalone unit test for Vishwadeep's GPU Inference Module...")
    engine = LocalVisionEngine()
    
    # Create a synthetic 640x640 frame to verify inference pipeline
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    metrics = engine.run_inference(dummy_frame)
    
    print("Test Execution Metrics:", metrics)