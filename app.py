from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow React frontend to access this API

# =========================================================
# LOAD MODEL ON STARTUP
# =========================================================

BASE_DIR = os.path.dirname(__file__)

# Correct path → backend/ml/RF_Model.pkl
MODEL_PATH = os.path.join(BASE_DIR, "ml", "RF_Model.pkl")

model = None

try:
    print("\n[INFO] Backend Starting...")

    # Check ml folder contents
    ml_folder = os.path.join(BASE_DIR, "ml")

    if os.path.exists(ml_folder):
        print("[INFO] Files inside ml folder:")
        print(os.listdir(ml_folder))
    else:
        print("[ERROR] ml folder not found!")

    print(f"[INFO] Looking for model at: {MODEL_PATH}")

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[✓] Model loaded successfully!")
    else:
        print("[X] Model file not found!")

except Exception as e:
    print(f"[X] Error loading model: {e}")

# =========================================================
# PREDICT API
# =========================================================

@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded. Please ensure RF_Model.pkl is inside backend/ml folder and restart server."
        }), 500

    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data provided"}), 400

        print("\n[INFO] Incoming Request Data:")
        print(data)

        # =====================================================
        # PARSE INPUTS
        # =====================================================

        age = int(data.get('age', 0))

        network_str = data.get('network', 'No')
        prior_auth_str = data.get('prior_auth', 'No')

        billing = float(data.get('billing', 0.0))
        delay = int(data.get('delay', 0))

        plan = data.get('plan', '')
        procedure = data.get('procedure', '')
        diagnosis = data.get('diagnosis', '')

        # Convert Yes/No → 1/0
        network = 1 if network_str.lower() == "yes" else 0
        prior_auth = 1 if prior_auth_str.lower() == "yes" else 0

        # =====================================================
        # CREATE FEATURE VECTOR
        # =====================================================

        try:
            columns = model.feature_names_in_
        except AttributeError:
            return jsonify({
                "error": "Model missing feature_names_in_. Please retrain model with feature names."
            }), 500

        user_data = pd.DataFrame(columns=columns)
        user_data.loc[0] = 0

        # =====================================================
        # NUMERIC FEATURES
        # =====================================================

        if 'patient_age_years' in user_data.columns:
            user_data.loc[0, 'patient_age_years'] = age

        if 'is_in_network' in user_data.columns:
            user_data.loc[0, 'is_in_network'] = network

        if 'prior_auth_required' in user_data.columns:
            user_data.loc[0, 'prior_auth_required'] = prior_auth

        if 'billed_amount_usd' in user_data.columns:
            user_data.loc[0, 'billed_amount_usd'] = billing

        if 'days_between_service_and_submission' in user_data.columns:
            user_data.loc[0, 'days_between_service_and_submission'] = delay

        # =====================================================
        # ONE-HOT FEATURES
        # =====================================================

        # PLAN
        plan_col = f"insurance_plan_type_{plan}"
        if plan_col in user_data.columns:
            user_data.loc[0, plan_col] = 1

        # PROCEDURE
        procedure_col = f"procedure_code_cpt_{procedure}"
        if procedure_col in user_data.columns:
            user_data.loc[0, procedure_col] = 1

        # DIAGNOSIS
        diagnosis_col = f"primary_diagnosis_code_icd10_{diagnosis}"
        if diagnosis_col in user_data.columns:
            user_data.loc[0, diagnosis_col] = 1

        print("\n[INFO] Prepared Feature Data:")
        print(user_data.head())

        # =====================================================
        # MAKE PREDICTION
        # =====================================================

        prob = model.predict_proba(user_data)[0][1] * 100

        # Random reason mapping
        reason_map = {
            1: "Missing Documentation",
            2: "Invalid Procedure Code",
            3: "Authorization Missing",
            4: "Policy Expired",
            5: "Duplicate Claim"
        }

        if prob >= 70:
            status = "DENIED"
            reason = reason_map[np.random.randint(1, 6)]

        elif prob >= 40:
            status = "RISK OF DENIAL"
            reason = reason_map[np.random.randint(1, 6)]

        else:
            status = "APPROVED"
            reason = "None (Claim Approved)"

        response = {
            "status": status,
            "probability": f"{round(prob, 2)}%",
            "reason": reason
        }

        print("\n[INFO] Prediction Response:")
        print(response)

        return jsonify(response)

    except Exception as e:

        print(f"[ERROR] Prediction Failed: {e}")

        return jsonify({
            "error": f"Prediction logic failed: {str(e)}"
        }), 500


# =========================================================
# ROOT ROUTE (Health Check)
# =========================================================

@app.route('/')
def home():
    return jsonify({
        "message": "AI Claim Prediction API Running"
    })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == '__main__':
    print("\n🚀 Starting Flask AI Server...")
    app.run(debug=True, port=5000)