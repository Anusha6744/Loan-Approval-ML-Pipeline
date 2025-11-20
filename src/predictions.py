import joblib
import pandas as pd


model = joblib.load("loan_model.pkl")
print("Model loaded successfully!")

test_df = pd.read_csv("data/loan.csv").head(10)


X_test = test_df.drop(columns=["Loan_Status"], errors='ignore')


predictions = model.predict(X_test)
print("Predictions:", predictions)


if "Loan_Status" in test_df.columns:
    from sklearn.metrics import accuracy_score
    y_test = test_df["Loan_Status"]
    print("Test Accuracy:", accuracy_score(y_test, predictions))
