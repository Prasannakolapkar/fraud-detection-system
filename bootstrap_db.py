import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from database import DatabaseManager
from api.auth import pwd_context

def bootstrap():
    db = DatabaseManager(db_path="./data/fraud_detection.db")
    admin_user = "admin"
    admin_pass = "admin123"
    
    import numpy as np
    dummy_embedding = np.random.randn(512).astype(np.float32)
    dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)
    
    print(f"Bootstrapping admin user: {admin_user}")
    hashed_pw = pwd_context.hash(admin_pass)
    db.upsert_user(admin_user, "0000", embedding=dummy_embedding, role="ADMIN", password_hash=hashed_pw)
    print("Done (Admin enrolled with demo biometrics).")

if __name__ == "__main__":
    bootstrap()
