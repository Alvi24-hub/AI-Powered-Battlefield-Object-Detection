# compare_clahe.py
import cv2
import numpy as np
import os
from main import ThermalImageProcessor

def compare_clahe_effects():
    """Quantitatively compare different CLAHE settings"""
    
    # Load or generate a test image
    test_img_path = "thermal_sample.jpg"
    if not os.path.exists(test_img_path):
        # Generate a complex thermal image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add multiple gradients and patterns
        for i in range(480):
            for j in range(640):
                val = int(30 + (i / 480) * 50 + (j / 640) * 30 + 
                         40 * np.sin(i/30) + 30 * np.cos(j/40))
                img[i, j] = [val, val, val]
        # Add some "hot spots"
        for _ in range(10):
            x, y = np.random.randint(50, 590), np.random.randint(50, 430)
            radius = np.random.randint(10, 40)
            intensity = np.random.randint(150, 230)
            cv2.circle(img, (x, y), radius, (intensity, intensity, intensity), -1)
        cv2.imwrite(test_img_path, img)
    
    with open(test_img_path, 'rb') as f:
        image_bytes = f.read()
    
    configs = [
        (2.0, 4, 4),
        (3.0, 8, 8),
        (4.0, 8, 8),
        (4.5, 4, 4),
    ]
    
    print("📊 QUANTITATIVE CLAHE COMPARISON")
    print("=" * 70)
    print(f"{'Config':>20} | {'Mean':>8} | {'Std Dev':>8} | {'Min':>8} | {'Max':>8} | {'Edge Density':>12}")
    print("-" * 70)
    
    results = []
    
    for clip, tx, ty in configs:
        processor = ThermalImageProcessor(clip_limit=clip, tile_grid_size=(tx, ty))
        processed, latency = processor.process_frame(image_bytes)
        
        if processed is not None:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Calculate statistics
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            min_val = np.min(gray)
            max_val = np.max(gray)
            
            # Edge density (higher = more contrast/edges visible)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size * 100
            
            results.append({
                "config": f"clip={clip}, tile={tx}x{ty}",
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "edge_density": edge_density,
                "latency": latency
            })
            
            print(f"{results[-1]['config']:>20} | {mean_val:>8.1f} | {std_val:>8.1f} | {min_val:>8.0f} | {max_val:>8.0f} | {edge_density:>11.2f}%")
    
    print("=" * 70)
    
    # Find best config
    best_edge = max(results, key=lambda x: x['edge_density'])
    fastest = min(results, key=lambda x: x['latency'])
    
    print(f"\n✅ Best contrast (edge density): {best_edge['config']} ({best_edge['edge_density']:.2f}%)")
    print(f"⚡ Fastest: {fastest['config']} ({fastest['latency']:.2f}ms)")
    
    # Recommendation
    print("\n💡 RECOMMENDATION:")
    if best_edge['config'] == fastest['config']:
        print(f"   Use: {best_edge['config']} (best contrast AND fastest)")
    else:
        print(f"   For quality: {best_edge['config']}")
        print(f"   For speed:   {fastest['config']}")
        print(f"   Balanced:    clip=3.0, tile=8x8 (default)")
    
    return results

if __name__ == "__main__":
    compare_clahe_effects()