from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.impute import SimpleImputer

def build_pipeline():
    categorical_cols = []
    numerical_cols = []
 
    sample_df = pd.read_csv("data/loan.csv").head(1)
    for col in sample_df.columns:
        if col in ["Loan_ID", "Loan_Status"]:
            continue
        if sample_df[col].dtype == "object":
            categorical_cols.append(col)
        elif sample_df[col].dtype in ("int64", "float64"):
            numerical_cols.append(col)

   

    preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols),
        
        ("num", SimpleImputer(strategy="mean"), numerical_cols)
    ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    return pipeline
