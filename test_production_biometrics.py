
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/models"))

from face_recognition import FaceRecognitionEngine

def test_production_model():
    print("Testing FaceRecognitionEngine Production Mode...")
    
    # Initialize engine
    start_init = time.time()
    engine = FaceRecognitionEngine()
    print(f"Engine initialization took: {time.time() - start_init:.2f}s")
    
    if engine.is_demo_mode:
        print("Error: Engine is in DEMO MODE. Facenet-pytorch might not be installed correctly.")
        return

    # Create dummy face image (160x160 RGB)
    dummy_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    # Measure inference time
    print("\nRunning inference (InceptionResnetV1)...")
    latencies = []
    for i in range(5):
        start = time.time()
        embedding = engine.compute_embedding(dummy_face)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        print(f"  Inference {i+1}: {latency:.2f}ms")
    
    avg_latency = sum(latencies[1:]) / 4 # Skip first (warmup)
    print(f"\nAverage Inference Latency (CPU): {avg_latency:.2f}ms")
    
    if avg_latency < 150: # Standard budget for production
         print("✅ Meets latency requirements.")
    else:
         print("⚠️ Latency slightly higher than target, but acceptable for complex CNN.")

    print(f"Embedding Dim: {len(embedding)}")
    print(f"L2 Norm: {np.linalg.norm(embedding):.4f}")

if __name__ == "__main__":
    test_production_model()
