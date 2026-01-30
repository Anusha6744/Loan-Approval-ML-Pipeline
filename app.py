from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
MODEL_PATH = 'models/loan_model_v5.pkl'
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    # add more features as needed

    prediction = model.predict([[feature1, feature2]])[0]
    return f"Prediction: {prediction}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
