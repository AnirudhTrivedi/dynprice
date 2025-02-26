import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load dataset
df = pd.read_csv("airline_booking_dataset.csv")

# Feature Selection
features = [
    "Traveler_Type", "Membership_Status", "Days_Before_Travel", "Peak_Season",
    "Base_Ticket_Price", "WiFi_Purchased", "Lounge_Access_Purchased", 
    "Meals_Purchased", "Insurance_Purchased", "Priority_Boarding_Purchased", 
    "Carbon_Offset_Purchased", "Total_Ancillary_Cost"
]

# Encode categorical variables
df = pd.get_dummies(df, columns=["Traveler_Type", "Membership_Status", "Peak_Season"], drop_first=True)

# Define target variable (WTP approximation using Final Ticket Price)
X = df[features]
y_reg = df["Final_Ticket_Price"]  # For regression
y_clf = (df["Final_Ticket_Price"] > df["Base_Ticket_Price"] * 1.2).astype(int)  # For classification (WTP > 120% of base price)

# Train-test split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, X_test_clf, _, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Train Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_reg)

# Predict WTP using Random Forest
y_pred_reg = rf_model.predict(X_test)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

# Train Logistic Regression for classification
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_clf)

# Predict WTP Probability
y_pred_clf = log_model.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)

# Predict WTP for a Sample Traveler
sample_traveler = pd.DataFrame({
    "Days_Before_Travel": [5],
    "Base_Ticket_Price": [500],
    "WiFi_Purchased": [1],
    "Lounge_Access_Purchased": [1],
    "Meals_Purchased": [1],
    "Insurance_Purchased": [0],
    "Priority_Boarding_Purchased": [1],
    "Carbon_Offset_Purchased": [0],
    "Total_Ancillary_Cost": [50],
    "Traveler_Type_Business": [1],
    "Membership_Status_Gold": [1],
    "Peak_Season_Yes": [1]
})

predicted_wtp = rf_model.predict(sample_traveler)[0]
predicted_wtp_class = log_model.predict(sample_traveler)[0]
predicted_wtp_probability = log_model.predict_proba(sample_traveler)[0][1]  # Probability of high WTP

# Save WTP predictions in JSON format
wtp_results = {
    "Regression_Model": {
        "Mean_Absolute_Error": mae,
        "Predicted_WTP": predicted_wtp
    },
    "Logistic_Regression_Model": {
        "Accuracy": accuracy,
        "Predicted_WTP_Class": "High" if predicted_wtp_class == 1 else "Low",
        "Predicted_WTP_Probability": predicted_wtp_probability
    }
}

with open("wtp_predictions.json", "w") as json_file:
    json.dump(wtp_results, json_file, indent=4)

print("WTP analysis completed! Results saved in 'wtp_predictions.json'.")
