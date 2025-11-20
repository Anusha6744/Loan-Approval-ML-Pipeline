import joblib
import pandas as pd
import os

def get_latest_model(models_dir="models", base_name="loan_model"):
    
    files = os.listdir(models_dir)
    versions = []

    for f in files:
        if f.startswith(base_name) and f.endswith(".pkl"):
            try:
                v = int(f.split("_v")[1].split(".pkl")[0])
                versions.append((v, f))
            except:
                continue

    if not versions:
        raise FileNotFoundError("No model versions found in the models directory.")

    
    latest_version_file = sorted(versions, key=lambda x: x[0])[-1][1]
    return os.path.join(models_dir, latest_version_file)


def load_model(model_path):
    return joblib.load(model_path)


def predict_loan_status(model, input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return prediction


model_path = get_latest_model()
print("Loaded model:", model_path)

model = load_model(model_path)

sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 4000,
        "CoapplicantIncome": 1200,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }

result = predict_loan_status(model, sample)
print("Loan Status Prediction:", result)
