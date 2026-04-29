import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from database import DatabaseManager
from api.auth import pwd_context

def update_users():
    db = DatabaseManager(db_path="data/fraud_detection.db")
    users = ["alice", "bob"]
    
    for uid in users:
        print(f"[INFO] Updating password for user: {uid}")
        hashed_pw = pwd_context.hash(uid + "123")
        db.upsert_user(
            user_id=uid,
            card_number="MIGRATED", # Preserve card dummy
            password_hash=hashed_pw,
            role="CARDHOLDER"
        )
    print(f"[SUCCESS] Updated {len(users)} users with default passwords (e.g., alice123).")

if __name__ == "__main__":
    update_users()
