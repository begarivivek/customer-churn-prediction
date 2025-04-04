import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Test input data
def test_prediction(age, gender, tenure, monthly_charges):
    # Convert gender to numeric (Female=1, Male=0)
    gender_selected = 1 if gender.lower() == "female" else 0

    # Prepare the input features
    X_input = [[age, gender_selected, tenure, monthly_charges]]

    # Convert to DataFrame (same as in training)
    feature_names = ["Age", "Gender", "Tenure", "MonthlyCharges"]
    X_df = pd.DataFrame(X_input, columns=feature_names)

    # Scale the input features using the scaler
    X_scaled = scaler.transform(X_df)

    # Get the prediction
    prediction = model.predict(X_scaled)[0]
    
    # Return "Yes" if churn (1), else "No"
    return "Yes" if prediction == 1 else "No"


# Example inputs to test
inputs = [
    (30, "Male", 12, 80),  # You can test different inputs here
    (45, "Female", 3, 90),
    (50, "Male", 5, 100),
    (25, "Female", 6, 70)
]

# Test the predictions
for input_data in inputs:
    age, gender, tenure, monthly_charges = input_data
    prediction = test_prediction(age, gender, tenure, monthly_charges)
    print(f"Prediction for Age: {age}, Gender: {gender}, Tenure: {tenure}, MonthlyCharges: {monthly_charges} -> Churn: {prediction}")
