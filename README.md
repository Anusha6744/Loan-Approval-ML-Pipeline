
#  Loan Approval Prediction – End-to-End ML Pipeline (with MLOps Concepts)

This project builds a **fully automated machine learning pipeline** to predict loan approval status using a CSV dataset.  
It includes:

* Automatic preprocessing  
* Missing value handling using SimpleImputer  
* Encoding using OneHotEncoder  
* Pipeline-based training  
* Hyperparameter tuning  
* Model versioning  
* A separate prediction script  
* Clean modular structure  


#  Project Features

## 1️) **Automated Column Detection**
The pipeline automatically reads the dataset and detects:

- **Categorical columns** → dtype = object  
- **Numerical columns** → dtype = int/float  

No need to manually list columns.

---

## 2️) **Automated Preprocessing Pipeline**

### * Categorical Columns  
Processed using a mini-pipeline:

| Step | Transformer | Purpose |
|------|-------------|---------|
| 1 | `SimpleImputer(strategy="most_frequent")` | Handles missing values |
| 2 | `OneHotEncoder(handle_unknown="ignore")` | Converts categories to numbers |

### * Numerical Columns  
Processed using:

| Transformer | Purpose |
|-------------|---------|
| `SimpleImputer(strategy="mean")` | Replaces missing numeric values |

### * Final Model  
`RandomForestClassifier(random_state=42)`

All combined inside:

```python
Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(...))
])
```

This ensures **no manual fit/transform**, **no data leakage**, and **full automation**.

---

## 3️) **Hyperparameter Tuning**
`GridSearchCV` is used to pick the best Random Forest parameters:

```
n_estimators = [50, 100, 150]
max_depth    = [5, 10, None]
```

The best model is returned as:

- `best_estimator_`
- `best_params_`

---

## 4️) **Model Versioning**
Every trained model is saved automatically as:

```
models/loan_model_v1.pkl
models/loan_model_v2.pkl
models/loan_model_v3.pkl
```

Ensures reproducibility & basic MLOps.

---

## 5️) **Prediction Script**
`main.py` loads **latest model version** and predicts loan status for new user inputs.

---

#  How the Pipeline Works 

### **1. Load data**
```python
df = load_data("data/loan.csv")
```

### **2. Split features & target**
```python
x, y = split_features_target(df, "Loan_Status")
```

### **3. Build preprocessing + model pipeline**
Automatically detects columns, imputes missing values, encodes categorical data.

### **4. Hyperparameter tuning**
Uses GridSearchCV (cv=3).

### **5. Train final model**
```python
best_pipeline.fit(x, y)
```
### **6. Testing and evaluation**
  • Accuracy
  • Confusion Matrix
  • Classification Report


### **7. Versioned saving**
Automatically creates:
```
models/loan_model_vX.pkl
```

### **8. Prediction**
Using:
```python
predict_loan_status(model, sample)
```

---

#  Folder Structure

```
Loan Project/
│── data/
│   └── loan.csv
│── models/
│── src/
│   ├── preprocess.py
│   ├── pipeline.py
│   ├── tuning.py
│   └── training.py
│── main.py
│── requirements.txt
│── README.md
│── .gitignore
│── venv/   (ignored)
```

---

#  Running the Project

### 1. Create virtual environment
```
python -m venv venv
```

### 2. Activate it
```
venv\Scripts\activate
```

### 3. Install requirements
```
pip install -r requirements.txt
```

### 4. Train model
```
python src/training.py
```

### 5. Make predictions
```
python main.py
```

---

#  Dependencies (requirements.txt)

```
pandas
numpy
scikit-learn
joblib
```

---

#  Why Random Forest?

- Handles categorical + numerical data  
- Handles missing values effectively  
- Low overfitting risk  
- Works well without feature scaling  
- Strong baseline performance for tabular datasets  

This makes it suitable for **loan approval classification**.

---

### This project demonstrates:

- End-to-end ML development  
- Automated preprocessing  
- Missing value handling  
- Hyperparameter tuning  
- Model versioning  
- Reusable prediction pipeline  
- Clean Git/GitHub workflow  
