import os
import re
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import your model classes from the models directory
from models.alticred_salaried import AltiCredScorer
from models.student import AdaptabilityScorer
from models.farmers import main as farmers_main, generate_synthetic_data

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- Load Models on App Startup ---
# Note: For production, you should pre-train and save your models to .pkl files
# and load them here to avoid long startup times.
print("Loading all models...")
salaried_scorer = AltiCredScorer(file_path='./data/salaried_dataset.csv')
student_scorer = AdaptabilityScorer(file_path='./data/modified_student_data_v2.csv')

# The farmers.py script needs a different approach
# We'll run its main function to train the models and then
# extract the final prediction logic.
generate_synthetic_data(file_path='./data/farmers_data.csv')
from models.farmers import main
main() # This will train the farmer models and save them
# Now, we would ideally load the saved models from `farmers.py`
# For this example, we'll assume we can use the logic directly
# after running the main function.
print("All models loaded successfully!")


# --- Helper Function for Natural Language Parsing ---
def parse_user_input(user_text, user_type):

    parsed_data = {}
    lower_text = user_text.lower()

    if user_type == "salaried":
        # Example parsing logic for salaried employee features
        sal_match = re.search(r"salary is (\d+)", lower_text)
        if sal_match:
            parsed_data["monthly_income"] = float(sal_match.group(1))

        credit_match = re.search(r"credit bills are (\d+)", lower_text)
        if credit_match:
            parsed_data["monthly_credit_bills"] = float(credit_match.group(1))

        # This is where you'd add more parsing for all 15 features...
        # For simplicity, we'll assume the user provides some data
        # and fill the rest with default values.
        
        # Default values for missing features (based on your dataset)
        defaults = {
            'connections': '', 'defaulter_neighbors': 0, 'verified_neighbors': 0,
            'monthly_credit_bills': 0, 'bnpl_utilization_rate': 0.05,
            'mortgage_months_left': 0, 'upi_balances': '[]', 'emi_status_log': '[]',
            'income-expense ratio': 1.1, 'owns_home': 0, 'monthly_rent': 0,
            'recovery_days': 15, 'mortgage_status': 'none', 'user_posts': '',
            'sentiment_score': 0.0
        }
        
        final_data = defaults.copy()
        final_data.update(parsed_data)
        return final_data

    # Add similar parsing logic for 'student' and 'farmer'
    elif user_type == "student":
        # ...
        return parsed_data
    elif user_type == "farmer":
        # ...
        return parsed_data
    
    return {}


def get_advice(score, user_type):
    if score >= 0.7:
        return "Your AltiCred Score is high! This indicates a strong likelihood of loan approval. Keep up the good work by maintaining a strong financial position."
    elif 0.5 <= score < 0.7:
        return "Your AltiCred Score is good, but there's room for improvement. Focus on reducing debt and increasing your income stability to further improve your score."
    else:
        # A more advanced version would use feature importance to give specific advice
        return "Your AltiCred Score is low. This suggests a lower likelihood of loan approval. To improve, you can focus on: reducing outstanding debts, increasing your income, and building a more stable financial history."


# --- API Endpoint to Get Score ---
@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_type = data.get('user_type')
    user_text = data.get('message')
    
    if not user_type or not user_text:
        return jsonify({"error": "Missing user_type or message"}), 400

    # Parse the user's message to get structured features
    user_features = parse_user_input(user_text, user_type)
    
    # Calculate the score based on the user type
    if user_type == "salaried":
        try:
            score = salaried_scorer.predict_alticred_score(user_features)
            advice = get_advice(score, user_type)
            return jsonify({"score": score, "advice": advice})
        except Exception as e:
            return jsonify({"error": f"Scoring failed for salaried: {e}"}), 500
    
    elif user_type == "student":
        # Placeholder for student model prediction
        try:
            score = student_scorer.predict(user_features)
            advice = get_advice(score, user_type)
            return jsonify({"score": score, "advice": advice})
        except Exception as e:
            return jsonify({"error": f"Scoring failed for student: {e}"}), 500
            
    elif user_type == "farmer":
        # Placeholder for farmer model prediction
        # The farmer model returns a score on a different scale, so we need to normalize it
        # Assume a range of 0-100 for the farmer score and normalize it.
        # This is an assumption as your script did not specify a range.
        try:
            # We would need to implement the farmer prediction logic here
            # For this example, let's use a dummy score
            score = np.random.uniform(0, 1)
            advice = get_advice(score, user_type)
            return jsonify({"score": score, "advice": advice})
        except Exception as e:
            return jsonify({"error": f"Scoring failed for farmer: {e}"}), 500
            
    else:
        return jsonify({"error": "Invalid user_type"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)