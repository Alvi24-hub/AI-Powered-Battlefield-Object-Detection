# test_resolutions.py
import cv2
import numpy as np
import time
from main import ThermalImageProcessor

def test_resolutions():
    """Test preprocessing with various image resolutions"""
    
    processor = ThermalImageProcessor(clip_limit=4.0, tile_grid_size=(8, 8))
    
    resolutions = [
        (320, 240),   # Low
        (640, 480),   # Standard
        (800, 600),   # Medium
        (1280, 720),  # HD
        (1920, 1080), # Full HD
        (3840, 2160), # 4K
    ]
    
    print("📊 RESOLUTION VALIDATION TEST")
    print("=" * 60)
    print(f"{'Resolution':>15} | {'Status':>10} | {'Latency (ms)':>12} | {'Size (KB)':>10}")
    print("-" * 60)
    
    results = []
    
    for width, height in resolutions:
        # Create test image
        test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', test_img)
        image_bytes = buffer.tobytes()
        
        # Process
        start_time = time.perf_counter()
        processed, latency = processor.process_frame(image_bytes)
        total_time = (time.perf_counter() - start_time) * 1000
        
        if processed is not None:
            status = "✅ PASS"
            size_kb = round(len(image_bytes) / 1024, 1)
        else:
            status = "❌ FAIL"
            size_kb = 0
        
        results.append({
            "width": width,
            "height": height,
            "status": status,
            "latency": total_time,
            "size_kb": size_kb
        })
        
        print(f"{width}x{height:>5} | {status:>10} | {total_time:>11.2f}ms | {size_kb:>10.1f}")
    
    print("=" * 60)
    
    # Summary
    passed = sum(1 for r in results if r["status"] == "✅ PASS")
    total = len(results)
    print(f"✅ {passed}/{total} resolutions passed")
    
    # Check for any resolution that caused issues
    failed = [r for r in results if r["status"] == "❌ FAIL"]
    if failed:
        # FIXED: Properly formatted f-string
        failed_list = [f"{r['width']}x{r['height']}" for r in failed]
        print(f"❌ Failed resolutions: {failed_list}")
    else:
        print("✅ All resolutions passed!")
    
    return results

if __name__ == "__main__":
    test_resolutions()