import joblib
from preprocess import load_data, split_features_target
from tuning import tune_hyperparameters
import os


def get_next_model_version(models_dir="models", base_name="loan_model"):
    
    os.makedirs(models_dir, exist_ok=True)
    existing_files = os.listdir(models_dir)
    print(existing_files)
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
    df = load_data("data/loan.csv")
    x_train, y_train = split_features_target(df, "Loan_Status")

    
   
    best_pipeline, best_params = tune_hyperparameters(x_train, y_train)
    print("Best Parameters:", best_params)

    best_pipeline.fit(x_train, y_train)

    

   
    model_path = get_next_model_version()
    joblib.dump(best_pipeline, model_path)
    print(f"Model saved as {model_path}")

train_model()
