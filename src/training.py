import joblib
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import load_data, split_features_target
from tuning import tune_hyperparameters


def get_next_model_version(models_dir="models", base_name="loan_model"):
    os.makedirs(models_dir, exist_ok=True)
    existing_files = os.listdir(models_dir)
    versions = []
    for f in existing_files:
        if f.startswith(base_name) and f.endswith(".pkl"):
            try:
                v = int(f.split("_v")[1].split(".pkl")[0])
                versions.append(v)
            except:
                continue
    next_version = max(versions, default=0) + 1
    return os.path.join(models_dir, f"{base_name}_v{next_version}.pkl")


def train_model():

    # 1. Load and prepare data
    df = load_data("data/loan.csv")
    X, y = split_features_target(df, "Loan_Status")

    X_train, y_train = X, y
    X_test, y_test = X, y

    # 2. Hyperparameter Tuning
    best_pipeline, best_params = tune_hyperparameters(X_train, y_train)
    best_pipeline.fit(X_train, y_train)

    # 3. Model Evaluation

    y_pred = best_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", clf_report)
    print("Confusion Matrix:\n", cm)

    # 4. MLflow Tracking

    mlflow.set_experiment("Loan Approval Project")

    with mlflow.start_run():

        # Log parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Save classification report as text file
        with open("classification_report.txt", "w") as f:
            f.write(clf_report)
        mlflow.log_artifact("classification_report.txt")

        # Save confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # Log model
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="Loan_Approval_Model"
        )

    
    # 5. Save versioned model locally
    model_path = get_next_model_version()
    joblib.dump(best_pipeline, model_path)
    print(f"Model saved as: {model_path}")


train_model()
