import requests
import json

BASE_URL = "http://localhost:8000"

def verify():
    print("--- 1. Testing Login ---")
    login_data = {"username": "admin", "password": "admin123"}
    res = requests.post(f"{BASE_URL}/login", data=login_data)
    if res.status_code == 200:
        token = res.json()["access_token"]
        print("[OK] Login successful.")
    else:
        print(f"[FAIL] Login failed: {res.text}")
        return

    print("\n--- 1b. Testing Invalid Login ---")
    res = requests.post(f"{BASE_URL}/login", data={"username": "admin", "password": "wrong"})
    print(f"Status: {res.status_code}")
    print(f"Response: {res.json()}")

if __name__ == "__main__":
    verify()
