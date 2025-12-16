from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("models/loan_model_v5.pkl")

@app.route("/")
def home():
    return "Loan Approval API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"loan_status": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
