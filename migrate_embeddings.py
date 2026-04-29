import json
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from database import DatabaseManager

def migrate():
    json_path = "data/face_embeddings/embeddings.json"
    db_path = "data/fraud_detection.db"
    
    if not os.path.exists(json_path):
        print(f"[INFO] No JSON file found at {json_path}. Nothing to migrate.")
        return

    print(f"[INFO] Migrating embeddings from {json_path} to {db_path}...")
    db = DatabaseManager(db_path=db_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    count = 0
    for user_id, embedding_list in data.items():
        embedding = np.array(embedding_list, dtype=np.float32)
        # We don't have card numbers in the JSON, using user_id as placeholder
        db.upsert_user(user_id, "MIGRATED", embedding)
        count += 1
    
    print(f"[SUCCESS] Migrated {count} users.")
    
    # Rename old file to avoid re-migration
    os.rename(json_path, json_path + ".bak")
    print(f"[INFO] Renamed {json_path} to {json_path}.bak")

if __name__ == "__main__":
    migrate()
