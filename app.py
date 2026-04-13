from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "RF_Model.pkl")

model = None

def load_model():
    global model
    try:
        print("🚀 Starting Backend...")

        if not os.path.exists(MODEL_PATH):
            print("❌ Model file not found:", MODEL_PATH)
            return

        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")

    except Exception as e:
        print("❌ Model loading failed:", str(e))

load_model()

# =========================================================
# ROOT ROUTE
# =========================================================

@app.route("/")
def home():
    return jsonify({
        "message": "AI Claim Prediction API Running 🚀",
        "status": "success"
    })

# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

# =========================================================
# PREDICT API
# =========================================================

@app.route('/api/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded. Check ml/RF_Model.pkl"
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # =============================
        # INPUT PARSING
        # =============================

        age = int(data.get('age', 0))
        network = 1 if data.get('network', 'No').lower() == "yes" else 0
        prior_auth = 1 if data.get('prior_auth', 'No').lower() == "yes" else 0
        billing = float(data.get('billing', 0))
        delay = int(data.get('delay', 0))

        plan = data.get('plan', '')
        procedure = data.get('procedure', '')
        diagnosis = data.get('diagnosis', '')

        # =============================
        # FEATURE CREATION
        # =============================

        try:
            columns = model.feature_names_in_
        except:
            return jsonify({
                "error": "Model missing feature_names_in_. Retrain model properly."
            }), 500

        user_data = pd.DataFrame(columns=columns)
        user_data.loc[0] = 0

        # Numeric mapping
        mapping = {
            'patient_age_years': age,
            'is_in_network': network,
            'prior_auth_required': prior_auth,
            'billed_amount_usd': billing,
            'days_between_service_and_submission': delay
        }

        for col, val in mapping.items():
            if col in user_data.columns:
                user_data.loc[0, col] = val

        # One-hot encoding mapping
        for col in user_data.columns:
            if col.startswith("insurance_plan_type_") and plan in col:
                user_data.loc[0, col] = 1

            if col.startswith("procedure_code_cpt_") and procedure in col:
                user_data.loc[0, col] = 1

            if col.startswith("primary_diagnosis_code_icd10_") and diagnosis in col:
                user_data.loc[0, col] = 1

        # =============================
        # PREDICTION
        # =============================

        prob = float(model.predict_proba(user_data)[0][1] * 100)

        if prob >= 70:
            status = "DENIED"
            reason = "High-risk claim pattern detected"

        elif prob >= 40:
            status = "RISK OF DENIAL"
            reason = "Moderate risk detected"

        else:
            status = "APPROVED"
            reason = "Low-risk claim"

        return jsonify({
            "status": status,
            "probability": f"{round(prob, 2)}%",
            "reason": reason
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# RUN SERVER (PRODUCTION SAFE)
# =========================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
