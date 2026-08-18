import cv2
import numpy as np
import time
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ------------------------- LOGGING SETUP -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Alvira] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------- FASTAPI APP -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preprocessing Engine starting up...")
    yield
    logger.info("Preprocessing Engine shutting down...")

app = FastAPI(
    title="Battlefield Vision - Preprocessing Engine",
    description="Thermal enhancement and frame decoding pipeline",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------- CORE PREPROCESSOR CLASS -------------------------
class ThermalImageProcessor:
    """
    Thermal Image Preprocessor using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances thermal signatures behind camouflage/foliage while preventing noise amplification.
    """
    
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)):
        """
        Args:
            clip_limit: Threshold for contrast limiting (higher = more contrast)
            tile_grid_size: Size of grid for histogram equalization (smaller = more local adaptation)
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        logger.info(f"CLAHE initialized | clip_limit={clip_limit}, tile_grid={tile_grid_size}")
    
    def process_frame(self, image_bytes: bytes) -> tuple:
        """
        Process a raw image byte buffer through the thermal enhancement pipeline.
        
        Steps:
        1. Decode JPEG bytes to BGR image matrix
        2. Convert to grayscale (CLAHE works on single-channel)
        3. Apply CLAHE for adaptive contrast enhancement
        4. Convert back to 3-channel BGR for model compatibility
        
        Returns:
            Tuple of (processed_frame, preprocessing_latency_ms)
            Returns (None, 0.0) on failure
        """
        start_time = time.perf_counter()
        
        # Step 1: Decode raw binary buffer to OpenCV BGR matrix
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.warning("Failed to decode image bytes — invalid or corrupted payload")
            return None, 0.0
        
        try:
            # Step 2: Convert to grayscale (CLAHE requires single-channel input)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Apply CLAHE adaptive histogram equalization
            enhanced_gray = self.clahe.apply(gray)
            
            # Step 4: Reconvert to 3-channel BGR for YOLO model compatibility
            processed_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            # Calculate preprocessing latency in milliseconds
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            logger.debug(f"Frame processed | Latency: {latency_ms}ms | Shape: {processed_frame.shape}")
            return processed_frame, latency_ms
            
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            return None, 0.0
    
    def process_frame_with_metadata(self, image_bytes: bytes) -> dict:
        """Extended version that returns additional metadata."""
        processed_frame, latency = self.process_frame(image_bytes)
        
        if processed_frame is None:
            return {
                "status": "error",
                "message": "Invalid or corrupted image payload",
                "preprocessing_latency_ms": 0.0,
                "frame_dimensions": None
            }
        
        return {
            "status": "success",
            "preprocessing_latency_ms": latency,
            "frame_dimensions": {
                "width": int(processed_frame.shape[1]),
                "height": int(processed_frame.shape[0]),
                "channels": int(processed_frame.shape[2])
            }
        }


# ------------------------- GLOBAL INSTANCE -------------------------
processor = ThermalImageProcessor(
    clip_limit=4.0,
    tile_grid_size=(8, 8)
)


# ------------------------- REST API ENDPOINTS -------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for pipeline monitoring."""
    return {
        "status": "ONLINE",
        "system_mode": "AIR_GAPPED_OFFLINE",
        "lead": "Alvira Mohammed",
        "module": "Preprocessing & Frame Ingestion",
        "config": {
            "clip_limit": processor.clip_limit,
            "tile_grid_size": processor.tile_grid_size
        }
    }


@app.post("/api/v1/vision/process-frame")
async def process_single_frame(file: UploadFile = File(...)):
    """
    REST endpoint for processing a single image frame.
    Accepts image upload and returns preprocessing metrics.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        contents = await file.read()
        result = processor.process_frame_with_metadata(contents)
        
        if result["status"] == "error":
            return {
                "status": "error",
                "message": result["message"],
                "preprocessing_latency_ms": 0.0
            }
        
        return {
            "status": "success",
            "preprocessing_latency_ms": result["preprocessing_latency_ms"],
            "frame_dimensions": result["frame_dimensions"]
        }
        
    except Exception as e:
        logger.error(f"REST API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")


@app.get("/api/v1/vision/config")
async def get_processor_config():
    """Get current processor configuration."""
    return {
        "clip_limit": processor.clip_limit,
        "tile_grid_size": processor.tile_grid_size
    }


# ------------------------- STANDALONE EXECUTION -------------------------
if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("🚀 ALVIRA MOHAMMED - PREPROCESSING ENGINE")
    logger.info("📡 Starting FastAPI server on 0.0.0.0:8001")
    logger.info("📋 API Docs: http://localhost:8001/docs")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

# Add this method to your ThermalImageProcessor class
def test_parameters(self, image_bytes: bytes, clip_values: list, tile_sizes: list):
    """
    Test multiple CLAHE parameter combinations and return comparison results.
    Use this to find the optimal settings for your thermal/IR images.
    """
    results = {}
    original_frame = None
    
    # Decode the image
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image payload"}
    
    original_frame = frame.copy()
    
    for clip in clip_values:
        for tile in tile_sizes:
            # Create a new CLAHE instance for each combination
            clahe = cv2.createCLAHE(
                clipLimit=clip,
                tileGridSize=(tile, tile)
            )
            
            # Process
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)
            processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Generate a key
            key = f"clip_{clip}_tile_{tile}x{tile}"
            
            # Store result (you can save images to disk for comparison)
            results[key] = {
                "clip_limit": clip,
                "tile_grid": (tile, tile),
                "processed_frame": processed  # or save to file
            }
    
    return results

def process_frame(self, image_bytes: bytes) -> tuple:
    """Process frame with input safeguards"""
    
    # SAFEGUARD 1: Check if input is None or empty
    if image_bytes is None:
        logger.error("Input is None")
        return None, 0.0
    
    if len(image_bytes) == 0:
        logger.error("Input is empty (0 bytes)")
        return None, 0.0
    
    # SAFEGUARD 2: Check minimum size (valid JPEG is at least ~50 bytes)
    if len(image_bytes) < 50:
        logger.warning(f"Input too small: {len(image_bytes)} bytes")
        return None, 0.0
    
    start_time = time.perf_counter()
    
    try:
        # SAFEGUARD 3: Safely decode
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        
        if len(np_arr) == 0:
            logger.error("Numpy array is empty")
            return None, 0.0
            
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # SAFEGUARD 4: Check if decode was successful
        if frame is None:
            logger.warning("Failed to decode image bytes")
            return None, 0.0
        
        # SAFEGUARD 5: Check minimum dimensions
        if frame.shape[0] < 16 or frame.shape[1] < 16:
            logger.warning(f"Image too small: {frame.shape}")
            return None, 0.0
        
        # SAFEGUARD 6: Check for all-black or all-white images (optional)
        mean_val = np.mean(frame)
        if mean_val < 5 or mean_val > 250:
            logger.warning(f"Image may be corrupted: mean value {mean_val:.2f}")
            # Still process but log it
        
        # Normal processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        processed_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        logger.debug(f"Frame OK | Shape: {processed_frame.shape} | Mean: {np.mean(processed_frame):.2f} | Latency: {latency_ms}ms")
        
        return processed_frame, latency_ms
        
    except cv2.error as e:
        logger.error(f"OpenCV error: {e}")
        return None, 0.0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, 0.0