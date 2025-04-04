from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the home route to display the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle form submissions and predict churn
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the HTML form
        age = int(request.form['age'])
        gender = 1 if request.form['gender'] == 'Female' else 0
        tenure = int(request.form['tenure'])
        monthlycharge = float(request.form['monthlycharge'])

        # Prepare the data for prediction
        input_data = pd.DataFrame([[age, gender, tenure, monthlycharge]], columns=["Age", "Gender", "Tenure", "MonthlyCharges"])

        # Scale the input data using the scaler
        input_scaled = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_scaled)[0]
        prediction = "Yes" if prediction == 1 else "No"

        # Return the prediction result
        return render_template('index.html', prediction_text=f"The churn prediction is: {prediction}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)


# to run this type---> python app.py