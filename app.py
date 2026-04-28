import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "balanceDiffOrig", "balanceDiffDest", "errorBalanceOrig", "errorBalanceDest",
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def encode_type(transaction_type):
    """One-hot encode transaction type (drop_first=True, baseline=CASH_IN)."""
    return {
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_DEBIT":    int(transaction_type == "DEBIT"),
        "type_PAYMENT":  int(transaction_type == "PAYMENT"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
    }


def engineer_features(amount, old_orig, new_orig, old_dest, new_dest):
    """Compute the four engineered balance features."""
    return {
        "balanceDiffOrig":  old_orig - new_orig,
        "balanceDiffDest":  new_dest - old_dest,
        "errorBalanceOrig": new_orig + amount - old_orig,
        "errorBalanceDest": old_dest + amount - new_dest,
    }


def generate_explanation(transaction_type, amount, old_orig, new_orig,
                         old_dest, new_dest, is_fraud, confidence):
    """Generate a human-readable explanation of the prediction."""
    reasons = []

    if transaction_type in ("TRANSFER", "CASH_OUT"):
        reasons.append(
            f"{transaction_type} transactions carry elevated fraud risk."
        )

    if amount > 200_000:
        reasons.append(
            f"The transaction amount (${amount:,.2f}) is unusually large."
        )
    elif amount > 50_000:
        reasons.append(
            f"The transaction amount (${amount:,.2f}) is above average."
        )

    balance_drop = old_orig - new_orig
    if balance_drop > 0 and abs(balance_drop - amount) < 1:
        reasons.append(
            "The sender's balance dropped exactly by the transaction amount — a common pattern in legitimate transfers."
        )
    elif old_orig > 0 and new_orig == 0:
        reasons.append(
            "The sender's account was completely drained to zero, which is a known fraud indicator."
        )

    if old_dest == 0 and new_dest > 0:
        reasons.append(
            "The recipient account had a zero balance before receiving funds — a flag seen in mule accounts."
        )

    if not reasons:
        if is_fraud:
            reasons.append(
                "The combination of transaction features matches known fraud patterns in the model's training data."
            )
        else:
            reasons.append(
                "The transaction parameters fall within normal ranges observed in the training data."
            )

    return " ".join(reasons)


def get_risk_label(is_fraud, confidence_pct):
    """Determine risk level and display color class."""
    if is_fraud:
        if confidence_pct >= 85:
            return "High Risk", "danger"
        elif confidence_pct >= 60:
            return "Medium Risk", "warning"
        else:
            return "Moderate Risk", "warning"
    else:
        if confidence_pct >= 85:
            return "Low Risk", "success"
        else:
            return "Review Recommended", "warning"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", transaction_types=TRANSACTION_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    errors = []

    # --- Parse inputs ---
    def parse_float(name, label):
        val = request.form.get(name, "").strip()
        if val == "":
            errors.append(f"{label} is required.")
            return None
        try:
            return float(val)
        except ValueError:
            errors.append(f"{label} must be a valid number.")
            return None

    step_raw = request.form.get("step", "").strip()
    if step_raw == "":
        errors.append("Step is required.")
        step = None
    else:
        try:
            step = int(float(step_raw))
            if step < 1:
                errors.append("Step must be a positive integer.")
                step = None
        except ValueError:
            errors.append("Step must be a valid integer.")
            step = None

    transaction_type = request.form.get("transaction_type", "").strip()
    if transaction_type not in TRANSACTION_TYPES:
        errors.append("Invalid transaction type selected.")

    amount    = parse_float("amount",    "Amount")
    old_orig  = parse_float("oldbalanceOrg",  "Sender Old Balance")
    new_orig  = parse_float("newbalanceOrig", "Sender New Balance")
    old_dest  = parse_float("oldbalanceDest", "Receiver Old Balance")
    new_dest  = parse_float("newbalanceDest", "Receiver New Balance")

    if errors:
        return render_template(
            "index.html",
            transaction_types=TRANSACTION_TYPES,
            errors=errors,
            form_data=request.form,
        )

    # --- Build feature vector ---
    type_enc = encode_type(transaction_type)
    eng      = engineer_features(amount, old_orig, new_orig, old_dest, new_dest)

    row = {
        "step":           step,
        "amount":         amount,
        "oldbalanceOrg":  old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        **type_enc,
        **eng,
    }

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    # --- Predict ---
    prediction = int(model.predict(df)[0])
    is_fraud   = prediction == 1

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(df)[0]
        confidence = float(proba[prediction]) * 100
    else:
        confidence = 100.0

    risk_label, risk_class = get_risk_label(is_fraud, confidence)
    explanation = generate_explanation(
        transaction_type, amount, old_orig, new_orig,
        old_dest, new_dest, is_fraud, confidence
    )

    result = {
        "is_fraud":        is_fraud,
        "label":           "Fraudulent Transaction" if is_fraud else "Legitimate Transaction",
        "confidence":      f"{confidence:.1f}",
        "risk_label":      risk_label,
        "risk_class":      risk_class,
        "explanation":     explanation,
        "transaction_type": transaction_type,
        "amount":          f"{amount:,.2f}",
    }

    return render_template(
        "index.html",
        transaction_types=TRANSACTION_TYPES,
        result=result,
        form_data=request.form,
    )


if __name__ == "__main__":
    app.run(debug=True)
