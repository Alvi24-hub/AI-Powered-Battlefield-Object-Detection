# test_clahe.py
import cv2
import numpy as np
from main import ThermalImageProcessor
import os

def test_thermal_image(image_path: str):
    """Test different CLAHE parameters on an image"""
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        print("🔄 Generating dummy thermal image for testing...")
        image_path = generate_dummy_thermal_image()
        
        if image_path is None:
            print("❌ Failed to generate dummy image. Exiting.")
            return None
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    print(f"📸 Testing image: {image_path}")
    print(f"📊 File size: {len(image_bytes)} bytes")
    print("-" * 50)
    
    # Test configurations
    configs = [
        (2.0, 4, 4, "Mild contrast, fine details"),
        (3.0, 8, 8, "Default (balanced)"),
        (4.0, 8, 8, "Stronger contrast"),
        (3.0, 12, 12, "Global contrast"),
        (4.5, 4, 4, "High contrast, fine detail"),
        (2.5, 6, 6, "Mid-range"),
    ]
    
    results = []
    best_latency = float('inf')
    best_config = None
    
    # Also save visual results for comparison
    os.makedirs("clahe_results", exist_ok=True)
    
    for clip, tx, ty, desc in configs:
        processor = ThermalImageProcessor(
            clip_limit=clip,
            tile_grid_size=(tx, ty)
        )
        
        processed, latency = processor.process_frame(image_bytes)
        
        if processed is not None:
            # Save the processed image for visual comparison
            output_path = f"clahe_results/clip_{clip}_tile_{tx}x{ty}.jpg"
            cv2.imwrite(output_path, processed)
            
            results.append({
                "clip_limit": clip,
                "tile_grid": f"{tx}x{ty}",
                "latency_ms": latency,
                "desc": desc,
                "shape": processed.shape,
                "saved_path": output_path
            })
            
            if latency < best_latency:
                best_latency = latency
                best_config = f"clip={clip}, tile={tx}x{ty}"
        else:
            results.append({
                "clip_limit": clip,
                "tile_grid": f"{tx}x{ty}",
                "latency_ms": "FAILED",
                "desc": desc,
                "saved_path": None
            })
    
    # Print results
    print("\n📊 CLAHE Test Results:")
    print("=" * 70)
    for r in results:
        if r['latency_ms'] != "FAILED":
            print(f"clip={r['clip_limit']:4.1f}, tile={r['tile_grid']:>6} → {r['latency_ms']:>6.2f}ms | {r['desc']}")
            print(f"   💾 Saved: {r['saved_path']}")
        else:
            print(f"clip={r['clip_limit']:4.1f}, tile={r['tile_grid']:>6} → ❌ FAILED | {r['desc']}")
    
    print("=" * 70)
    if best_config:
        print(f"✅ Fastest config: {best_config} ({best_latency:.2f}ms)")
    
    print(f"\n📁 All processed images saved in 'clahe_results' folder")
    print(f"   Open the folder to visually compare the results!")
    
    return results

def generate_dummy_thermal_image():
    """Generate a dummy thermal-like image for testing"""
    print("🔥 Generating dummy thermal image for testing...")
    
    # Create a dark image (like thermal/IR)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a dark background with slight gradient (like sky/terrain)
    for i in range(480):
        for j in range(640):
            # Create a subtle gradient
            val = int(30 + (i / 480) * 40 + (j / 640) * 20)
            dummy[i, j] = [val, val, val]
    
    # Add some hot spots (like thermal signatures of vehicles/people)
    hot_spots = [
        (150, 200, 50, 180, 220),  # Large vehicle
        (350, 150, 30, 140, 190),  # Medium vehicle
        (500, 300, 25, 160, 210),  # Another vehicle
        (200, 350, 15, 130, 170),  # Person/small object
        (420, 250, 20, 150, 200),  # Another object
    ]
    
    for x, y, radius, min_intensity, max_intensity in hot_spots:
        intensity = np.random.randint(min_intensity, max_intensity)
        cv2.circle(dummy, (x, y), radius, (intensity, intensity, intensity), -1)
        # Add slight blur to make it look more realistic
        cv2.GaussianBlur(dummy, (5, 5), 1, dst=dummy)
    
    # Add some thermal noise
    noise = np.random.randint(0, 15, (480, 640, 3), dtype=np.uint8)
    dummy = cv2.add(dummy, noise)
    
    # Add some faint heat signatures (harder to detect)
    for _ in range(5):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        radius = np.random.randint(5, 15)
        intensity = np.random.randint(80, 120)
        cv2.circle(dummy, (x, y), radius, (intensity, intensity, intensity), -1)
    
    # Save to file
    temp_path = "thermal_sample.jpg"
    cv2.imwrite(temp_path, dummy)
    print(f"✅ Dummy thermal image saved to: {temp_path}")
    print(f"   This image simulates a thermal/IR scene with multiple targets\n")
    
    return temp_path

def test_clahe_parameter_sweep():
    """Run a comprehensive parameter sweep"""
    print("=" * 70)
    print("🔬 CLAHE THERMAL IMAGE TUNING")
    print("=" * 70)
    
    # Check if real image exists, otherwise generate dummy
    real_image = "thermal_sample.jpg"
    if not os.path.exists(real_image):
        real_image = generate_dummy_thermal_image()
    
    if real_image:
        test_thermal_image(real_image)

def quick_test():
    """Quick test with a single image and default settings"""
    print("=" * 70)
    print("⚡ QUICK TEST: Default CLAHE Settings")
    print("=" * 70)
    
    # Generate image if needed
    image_path = "thermal_sample.jpg"
    if not os.path.exists(image_path):
        image_path = generate_dummy_thermal_image()
    
    if not image_path:
        print("❌ No image available")
        return
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Test with default settings
    processor = ThermalImageProcessor(clip_limit=3.0, tile_grid_size=(8, 8))
    processed, latency = processor.process_frame(image_bytes)
    
    if processed is not None:
        # Save result
        cv2.imwrite("clahe_results/default_output.jpg", processed)
        print(f"✅ Image processed!")
        print(f"   Latency: {latency:.2f}ms")
        print(f"   Shape: {processed.shape}")
        print(f"   Saved to: clahe_results/default_output.jpg")
    else:
        print("❌ Processing failed!")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_test()
        elif sys.argv[1] == "--sweep":
            test_clahe_parameter_sweep()
        else:
            test_thermal_image(sys.argv[1])
    else:
        # Default: run parameter sweep
        test_clahe_parameter_sweep()