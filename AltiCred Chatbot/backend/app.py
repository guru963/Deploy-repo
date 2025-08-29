import os
import re
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Load environment variables from .env file
load_dotenv()

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import your model classes from the models directory
# Note: These files must exist and be correct
from models.alticred_salaried import AltiCredScorer
from models.student import AdaptabilityScorer
from models.farmers import main as farmers_main, generate_synthetic_data

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- Configure logging for production ---
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# --- Load Models on App Startup ---
app.logger.info("Loading all models...")
try:
    salaried_scorer = AltiCredScorer(file_path='./data/salaried_dataset.csv')
    student_scorer = AdaptabilityScorer(file_path='./data/modified_student_data_v2.csv')
    # The farmers.py script needs a different approach
    generate_synthetic_data(file_path='./data/farmers_data.csv')
    from models.farmers import main
    main()
    app.logger.info("All models loaded successfully!")
except Exception as e:
    app.logger.error(f"Failed to load models: {e}")
    # Consider if you want to exit or let the app run in a degraded state
    
# --- Configure Gemini API ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# --- Helper Function for Natural Language Parsing ---
def parse_user_input(user_text, user_type):
    parsed_data = {}
    lower_text = user_text.lower()

    if user_type == "salaried":
        sal_match = re.search(r"salary is (\d+)", lower_text)
        if sal_match:
            parsed_data["monthly_income"] = float(sal_match.group(1))

        credit_match = re.search(r"credit bills are (\d+)", lower_text)
        if credit_match:
            parsed_data["monthly_credit_bills"] = float(credit_match.group(1))
        
        defaults = {
            'connections': '', 'defaulter_neighbors': 0, 'verified_neighbors': 0,
            'bnpl_utilization_rate': 0.05,
            'mortgage_months_left': 0, 'upi_balances': '[]', 'emi_status_log': '[]',
            'income-expense ratio': 1.1, 'owns_home': 0, 'monthly_rent': 0,
            'recovery_days': 15, 'mortgage_status': 'none', 'user_posts': '',
            'sentiment_score': 0.0
        }
        
        final_data = defaults.copy()
        final_data.update(parsed_data)
        return final_data

    elif user_type == "student":
        # Add student parsing logic here
        return parsed_data
    elif user_type == "farmer":
        # Add farmer parsing logic here
        return parsed_data
    
    return {}

def get_advice(score, user_type):
    if score >= 0.7:
        return "Your AltiCred Score is high! This indicates a strong likelihood of loan approval. Keep up the good work by maintaining a strong financial position."
    elif 0.5 <= score < 0.7:
        return "Your AltiCred Score is good, but there's room for improvement. Focus on reducing debt and increasing your income stability to further improve your score."
    else:
        return "Your AltiCred Score is low. This suggests a lower likelihood of loan approval. To improve, you can focus on: reducing outstanding debts, increasing your income, and building a more stable financial history."

# --- Helper Function for Translation ---
def translate_to_english(text):
    prompt = f"Translate the following text to English, just give me the translated text, no extra phrases: '{text}'"
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        app.logger.error(f"Translation failed: {e}")
        return text

# --- API Endpoint to Get Score ---
@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_type = data.get('user_type')
    user_text = data.get('message')
    
    if not user_type or not user_text:
        return jsonify({"error": "Missing user_type or message"}), 400

    # Translate the user's message to English before parsing
    translated_text = translate_to_english(user_text)

    # Parse the user's translated message to get structured features
    user_features = parse_user_input(translated_text, user_type)
    
    # Calculate the score based on the user type
    if user_type == "salaried":
        try:
            score = salaried_scorer.predict_alticred_score(user_features)
            advice = get_advice(score, user_type)
            return jsonify({"score": score, "advice": advice})
        except Exception as e:
            return jsonify({"error": f"Scoring failed for salaried: {e}"}), 500
    
    elif user_type == "student":
        try:
            score = student_scorer.predict(user_features)
            advice = get_advice(score, user_type)
            return jsonify({"score": score, "advice": advice})
        except Exception as e:
            return jsonify({"error": f"Scoring failed for student: {e}"}), 500
            
    elif user_type == "farmer":
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
