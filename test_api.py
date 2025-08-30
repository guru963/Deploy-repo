# test_api.py
import requests
import json

# Test data with all required features
test_data = {
    "social_connections_count": 25,
    "proximity_to_defaulters_score": 0.2,
    "digital_network_engagement_value": 0.7,
    "support_request_frequency": 0.3,
    "market_access_value": 0.6,
    "time_to_resume_upi_after_shock": 5,
    "emi_status_last_12_months": 2,
    "overdraft_usage_frequency": 0.4,
    "loan_repayments_done": 12,
    "yield_recovered_units": 85,
    "pm_kisan_installments_received": 3,
    "income_volatility_value": 0.3,
    "budgeting_habit_value": 0.8,
    "agritech_tool_usage": 0.6,
    "new_crop_adoption_flag": 1
}

# Add optional features if your model expects them
optional_features = {
    "reliable_contacts_count": 8,
    "in_cooperative": 1,
    "utilizing_coop_benefit": 0.7
}

# Add optional features to test data if they're in your required features list
for feature in optional_features:
    if feature in test_data:  # This won't add them, but if you want to check which are required
        test_data[feature] = optional_features[feature]

url = "http://localhost:5000/predict"
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, data=json.dumps(test_data), headers=headers)
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", e)