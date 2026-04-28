# Fraud Detection System

A polished, production-ready Flask web application that uses a trained Random Forest model to assess transaction fraud risk from PaySim financial data.

---

## Features

- Clean fintech-style UI with dark/ivory aesthetics
- Real-time fraud prediction with confidence scores
- Risk level classification: High / Medium / Low / Review
- Human-readable explanation of the model's decision
- Full input validation with friendly error messages
- Responsive design for desktop and mobile

---

## Project Structure

```
fraud_app/
├── app.py                  # Flask backend
├── fraud_model.pkl         # Trained model (you provide this)
├── requirements.txt        # Python dependencies
├── README.md
├── templates/
│   └── index.html          # Main UI template
└── static/
    └── style.css           # Stylesheet
```

---

## Setup Instructions

### 1. Clone / place the project files

Make sure all files are in the same directory (e.g. `fraud_app/`).

### 2. Place your model file

Copy your trained model into the root of the project directory:

```
fraud_app/fraud_model.pkl
```

The app expects the model at the same level as `app.py`.

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
python app.py
```

The app will start at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Model Details

The model (`fraud_model.pkl`) is a Random Forest classifier trained on the [PaySim](https://www.kaggle.com/ntnu-testimon/paysim1) synthetic financial dataset.

### Feature Engineering (applied at inference time)

| Feature            | Formula                              |
|--------------------|--------------------------------------|
| `balanceDiffOrig`  | `oldbalanceOrg - newbalanceOrig`     |
| `balanceDiffDest`  | `newbalanceDest - oldbalanceDest`    |
| `errorBalanceOrig` | `newbalanceOrig + amount - oldbalanceOrg` |
| `errorBalanceDest` | `oldbalanceDest + amount - newbalanceDest` |

### Transaction Type Encoding

One-hot encoded with `drop_first=True`. Baseline category = `CASH_IN`.

Expected dummy columns: `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`.

---

## Input Fields

| Field                | Description                        |
|----------------------|------------------------------------|
| Step                 | Time unit in simulation (1 = 1 hr) |
| Transaction Type     | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Amount               | Value of the transaction           |
| Sender Old Balance   | Sender's balance before transaction |
| Sender New Balance   | Sender's balance after transaction  |
| Receiver Old Balance | Receiver's balance before transaction |
| Receiver New Balance | Receiver's balance after transaction |

---

## Risk Classification Logic

| Prediction | Confidence  | Risk Level          |
|------------|-------------|---------------------|
| Fraud      | ≥ 85%       | High Risk           |
| Fraud      | 60–84%      | Medium Risk         |
| Fraud      | < 60%       | Moderate Risk       |
| Legit      | ≥ 85%       | Low Risk            |
| Legit      | < 85%       | Review Recommended  |

---

## Notes

- No database or authentication required
- Model is loaded once at app startup for performance
- Debug mode is enabled by default; disable for production
