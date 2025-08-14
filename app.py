# app.py

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import sys

# Add the models directory to the Python path
sys.path.insert(0, './models')

# Import your scorer classes
from alticred_salaried import AltiCredScorer as SalariedScorer
# from alticred_self_employed import AltiCredScorer as SelfEmployedScorer # <-- Uncomment and adapt for your files
# from alticred_student import AltiCredScorer as StudentScorer           # <-- Uncomment and adapt for your files

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load Models on Startup ---
# This is efficient because models are loaded once, not on every request.
print("Loading models... Please wait.")
# We assume the CSVs are in a location accessible by the scripts.
# If not, you might need to adjust file paths inside your model classes.
salaried_scorer = SalariedScorer()
# self_employed_scorer = SelfEmployedScorer() # <-- Uncomment
# student_scorer = StudentScorer()             # <-- Uncomment
print("All models loaded successfully!")


# --- Define a Route for the Frontend ---
# This route will serve our main HTML page.
@app.route('/')
def home():
    """Renders the main user interface page."""
    return render_template('index2.html')


# --- Define a Route for Scoring ---
# This is our API endpoint. The frontend will send data here.
@app.route('/predict', methods=['POST'])
def predict():
    """Receives user data, selects a model, and returns a score."""
    try:
        # Get the JSON data sent by the frontend
        data = request.get_json()
        model_type = data.get('model_type')
        user_input = data.get('user_data')

        if not model_type or not user_input:
            return jsonify({'error': 'Missing model_type or user_data'}), 400

        # --- Data Type Conversion ---
        # HTML forms send everything as strings, so we must convert them back to numbers.
        numeric_fields = [
            'defaulter_neighbors', 'verified_neighbors', 'monthly_credit_bills',
            'bnpl_utilization_rate', 'mortgage_months_left', 'income-expense ratio',
            'owns_home', 'monthly_rent', 'recovery_days', 'sentiment_score'
        ]
        for field in numeric_fields:
            if field in user_input:
                user_input[field] = float(user_input[field])

        # --- Select the Model and Predict ---
        score = -1
        if model_type == 'salaried':
            score = salaried_scorer.predict_alticred_score(user_input)
        # elif model_type == 'self_employed':
        #     score = self_employed_scorer.predict_alticred_score(user_input) # <-- Uncomment
        # elif model_type == 'student':
        #     score = student_scorer.predict_alticred_score(user_input)     # <-- Uncomment
        else:
            return jsonify({'error': 'Invalid model type specified'}), 400

        # Return the result as a JSON object
        return jsonify({'alticred_score': f"{score:.4f}"})

    except Exception as e:
        # Return a more informative error
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Setting host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5001, debug=True)