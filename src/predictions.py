import joblib
import pandas as pd


model = joblib.load("loan_model.pkl")
print("Model loaded successfully!")

test_df = pd.read_csv("data/loan.csv").head(10)


X_test = test_df.drop(columns=["Loan_Status"], errors='ignore')


y_pred = model.predict(X_test)
print("Predictions:", y_pred)


if "Loan_Status" in test_df.columns:
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    y_test = test_df["Loan_Status"]
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:",classification_report(y_test,y_pred))
    print("Confusion matrix:",confusion_matrix(y_test,y_pred))

