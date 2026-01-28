from turtle import pd
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "models/loan_model_v5.pkl"  # adjust if needed
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return "Loan Approval API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({"loan_status": str(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
