"""
Streamlit UI - Card Fraud Detection Dashboard
Credit & Debit Card Fraud Detection System
Authors: Karan Sumbe, Isha Ghokane, Shantanu Aptikar, Shreya Pawar
Guide: Prof. Aradhana Pawar & Dr. Lakshmikant Malphedwar

Run with: streamlit run ui/app.py
"""

import sys
import os
import numpy as np
import requests
import json
import cv2
import time
from datetime import datetime
from PIL import Image

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    from models.fraud_classifier import FraudClassifier
    from models.face_recognition import FaceRecognitionEngine
    from database import DatabaseManager

    # Page config
    st.set_page_config(
        page_title="Card Fraud Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        color: white;
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .approved { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; }
    .rejected { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; }
    .review   { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0">🛡️ Card Fraud Detection System</h1>
        <p style="margin:5px 0 0 0; opacity:0.85">CNN Biometric + ML Transaction Analysis | Real-Time Protection</p>
    </div>
    """, unsafe_allow_html=True)

    # ── AUTHENTICATION HELPERS ──────────────────────────────────────
    def login_user(username, password):
        try:
            response = requests.post(
                "http://localhost:8000/login",
                data={"username": username, "password": password},
                timeout=5
            )
            if response.status_code == 200:
                st.session_state.token = response.json()["access_token"]
                st.session_state.username = username
                st.success("Logged in successfully!")
                return True
            else:
                st.error("Invalid credentials")
                return False
        except Exception as e:
            st.error(f"Login failed: {e}")
            return False

    def get_auth_header():
        if "token" in st.session_state:
            return {"Authorization": f"Bearer {st.session_state.token}"}
        return {}

    # ── SIDEBAR: LOGIN ──────────────────────────────────────────────
    with st.sidebar:
        st.title("🔐 Security Portal")
        if "token" not in st.session_state:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if login_user(u, p):
                        st.rerun()
        else:
            st.write(f"Logged in as: **{st.session_state.username}**")
            if st.button("Logout"):
                del st.session_state.token
                del st.session_state.username
                st.rerun()


    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager(db_path="./data/fraud_detection.db")

    if "classifier" not in st.session_state:
        with st.spinner("Loading ML classifier..."):
            clf = FraudClassifier(model_path="./models/")
            if not clf.load_model():
                clf.train(use_synthetic=True)
                clf.save_model()
        st.session_state.classifier = clf

    if "face_engine" not in st.session_state:
        st.session_state.face_engine = FaceRecognitionEngine(
            db_manager=st.session_state.db_manager
        )

    if "tx_history" not in st.session_state:
        st.session_state.tx_history = []

    # ── SIDEBAR ────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("", ["🔄 Process Transaction", "👤 Enroll Cardholder",
                              "📊 Dashboard & Analytics", "⚙️ System Settings"])

        st.divider()
        st.markdown("**System Status**")
        enrolled = st.session_state.face_engine.get_enrolled_users()
        st.success(f"✅ ML Model: Ready")
        st.success(f"✅ Face Engine: Ready")
        st.success(f"✅ Database: Connected")
        st.info(f"👥 Enrolled Users: {len(enrolled)}")
        st.info(f"📝 Transactions: {len(st.session_state.db_manager.get_history())}")

    # ── PAGE: PROCESS TRANSACTION ──────────────────────────────────
    if "Process Transaction" in page:
        st.subheader("🔄 Process Transaction")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### Transaction Details")
            user_id = st.selectbox("Cardholder", enrolled if enrolled else ["(no users enrolled)"])
            amount = st.number_input("Amount ($)", min_value=0.01, value=85.00, step=0.01)
            merchant_category = st.selectbox("Merchant Category",
                ["grocery", "restaurant", "gas", "utility", "electronics", "travel", "gambling", "crypto", "other"])
            is_online = st.checkbox("Card-Not-Present (Online)")
            is_foreign = st.checkbox("Foreign Transaction")

        with col2:
            st.markdown("##### Identity Verification")
            captured_img = st.camera_input("Smile for the camera!")
            
            st.markdown("##### Risk Context")
            hour = st.slider("Transaction Hour", 0, 23, datetime.now().hour)
            dist_home = st.number_input("Distance from Home (km)", 0.0, 10000.0, 5.0)
            dist_last = st.number_input("Distance from Last Tx (km)", 0.0, 5000.0, 3.0)
            daily_count = st.number_input("Transactions Today", 1, 50, 2)
            weekly_avg = st.number_input("Weekly Avg Daily Spend ($)", 1.0, 5000.0, 120.0)
            device_match = st.checkbox("Known Device", value=True)
            velocity_flag = st.checkbox("Rapid Succession Flag")

        if st.button("🔍 Analyze Transaction", type="primary", disabled=not enrolled or not captured_img):
            tx = {
                "amount": amount, "hour": hour, "day_of_week": datetime.now().weekday(),
                "merchant_category": merchant_category, "distance_from_home": dist_home,
                "distance_from_last_tx": dist_last, "time_since_last_tx": 120,
                "daily_tx_count": daily_count, "daily_spend": amount * daily_count,
                "weekly_avg_spend": weekly_avg, "is_foreign": is_foreign,
                "device_match": device_match, "is_online": is_online, "velocity_flag": velocity_flag
            }

            with st.spinner("Running dual-layer analysis..."):
                # Decode Streamlit image
                img = Image.open(captured_img)
                img_array = np.array(img.convert('RGB'))
                img_normalized = cv2.resize(img_array, (160, 160)).astype(np.float32) / 255.0

                # Gate 1: Face verification
                face_result = st.session_state.face_engine.verify_user(
                    user_id,
                    img_normalized
                )
                # Gate 2: ML scoring
                ml_result = st.session_state.classifier.predict(tx)
                
                # Gate 3: API Backend Update (Secured)
                try:
                    res = requests.post(
                        "http://localhost:8000/transaction",
                        json={
                            "user_id": user_id,
                            "card_number": "UNKNOWN", # In real app, fetched from session/enrollment
                            "amount": amount,
                            **tx
                        },
                        headers=get_auth_header(),
                        timeout=5
                    )
                except Exception:
                    pass

            st.divider()
            st.markdown("#### Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Face Similarity", f"{face_result.get('similarity_score', 0):.3f}", "Gate 1")
            c2.metric("Fraud Score", f"{ml_result['fraud_score']:.4f}", "Gate 2")
            c3.metric("Risk Level", ml_result["risk_level"])

            face_passed = face_result.get("match", False)
            if not face_passed:
                final = "BLOCKED — IDENTITY MISMATCH"
                st.markdown(f'<div class="rejected">🚫 <strong>{final}</strong><br>Face verification failed. Transaction blocked.</div>', unsafe_allow_html=True)
            elif ml_result["decision"] == "APPROVE":
                final = "APPROVED"
                st.markdown(f'<div class="approved">✅ <strong>{final}</strong><br>Transaction approved. Identity and behavior verified.</div>', unsafe_allow_html=True)
            elif ml_result["decision"] == "HOLD_FOR_REVIEW":
                final = "HELD FOR REVIEW"
                st.markdown(f'<div class="review">⚠️ <strong>{final}</strong><br>Flagged for manual review. Cardholder notified.</div>', unsafe_allow_html=True)
            else:
                final = "BLOCKED — FRAUD DETECTED"
                st.markdown(f'<div class="rejected">🚫 <strong>{final}</strong><br>High fraud probability. Transaction blocked and alert sent.</div>', unsafe_allow_html=True)

            # Add to history (DB handled via pipeline logic, but here we update UI local view if needed)
            st.session_state.db_manager.log_transaction({
                "tx_id": f"UI-{int(time.time())}",
                "user_id": user_id,
                "amount": amount,
                "decision": final,
                "fraud_score": ml_result["fraud_score"],
                "face_score": face_result.get("similarity_score", 0),
                "latency_ms": face_result.get('processing_time_ms', 0) + ml_result.get('latency_ms', 0),
                "timestamp": datetime.now().isoformat()
            })

    # ── PAGE: ENROLL ────────────────────────────────────────────────
    elif "Enroll" in page:
        st.subheader("👤 Enroll New Cardholder")
        st.info("In production, face capture uses a webcam. Demo mode simulates enrollment.")

        with st.form("enroll_form"):
            new_uid = st.text_input("User ID (e.g. alice_123)")
            card_num = st.text_input("Card Number (last 4 digits used)", max_chars=16)
            submitted = st.form_submit_button("Enroll User")

        if submitted and new_uid:
            fake_face = np.random.rand(160, 160, 3).astype(np.float32)
            success = st.session_state.face_engine.enroll_user(new_uid, fake_face)
            if success:
                # Update card number in DB
                st.session_state.db_manager.upsert_user(new_uid, card_num, fake_face)
                st.success(f"✅ User '{new_uid}' enrolled successfully!")
                st.info("Face embedding and card details securely stored in database.")
            else:
                st.error("Enrollment failed. Please try again.")

        st.divider()
        st.markdown("**Currently Enrolled Users:**")
        if enrolled:
            for uid in enrolled:
                st.write(f"  • {uid}")
        else:
            st.write("No users enrolled yet.")

    # ── PAGE: DASHBOARD ─────────────────────────────────────────────
    elif "Dashboard" in page:
        st.subheader("📊 Analytics Dashboard")

        history = st.session_state.db_manager.get_history()
        if not history:
            st.info("No transactions processed yet. Go to 'Process Transaction' to get started.")
        else:
            stats = st.session_state.db_manager.get_stats()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", stats['total_transactions'])
            c2.metric("Approved", stats['approved'])
            c3.metric("Blocked", stats['blocked'])
            c4.metric("Avg Latency", f"{stats['avg_latency_ms']}ms")

            st.divider()
            st.markdown("**Recent Transactions**")
            import pandas as pd
            df = pd.DataFrame(history)
            st.dataframe(df, use_container_width=True)

    # ── PAGE: SETTINGS ──────────────────────────────────────────────
    elif "Settings" in page:
        st.subheader("⚙️ System Settings")
        clf = st.session_state.classifier

        st.markdown("**Model Configuration**")
        new_fraud_threshold = st.slider("Fraud Rejection Threshold", 0.5, 0.95,
                                         clf.fraud_threshold, 0.01)
        new_review_threshold = st.slider("Hold-for-Review Threshold", 0.3, 0.6,
                                          clf.review_threshold, 0.01)
        new_face_threshold = st.slider("Face Match Threshold", 0.7, 0.95,
                                        st.session_state.face_engine.MATCH_THRESHOLD, 0.01)

        if st.button("Apply Settings"):
            clf.fraud_threshold = new_fraud_threshold
            clf.review_threshold = new_review_threshold
            st.session_state.face_engine.MATCH_THRESHOLD = new_face_threshold
            st.success("Settings updated successfully.")

        st.divider()
        if st.button("🔄 Retrain Model (Synthetic Data)"):
            with st.spinner("Retraining model..."):
                metrics = clf.train(use_synthetic=True)
                clf.save_model()
            st.success(f"Model retrained! Accuracy: {metrics['accuracy']:.3f} | AUC: {metrics['auc_roc']:.3f}")

        st.divider()
        st.markdown("**System Information**")
        st.json({
            "model_type": clf.model_type,
            "fraud_threshold": clf.fraud_threshold,
            "review_threshold": clf.review_threshold,
            "face_match_threshold": st.session_state.face_engine.MATCH_THRESHOLD,
            "model_trained": clf.is_trained,
            "demo_mode": st.session_state.face_engine.is_demo_mode,
            "enrolled_users": len(enrolled)
        })

else:
    print("Streamlit not installed. Run: pip install streamlit")
    print("Then launch UI with: streamlit run ui/app.py")
