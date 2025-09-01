from flask import Blueprint, request, jsonify, render_template
import sys

# Add the models directory to the Python path
sys.path.insert(0, './models')

# Import your scorer classes
from AltiCred.alticred_salaried import AltiCredScorer as SalariedScorer

salaried_bp = Blueprint('salaried', __name__, template_folder='templates')

print("[Salaried] Loading models... Please wait.")
salaried_scorer = SalariedScorer()
print("[Salaried] Model loaded successfully!")

def safe_float(value):
    try:
        if str(value).lower() in ['true', 'yes']:
            return 1.0
        elif str(value).lower() in ['false', 'no']:
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0

@salaried_bp.route('/')
def home():
    return render_template('index2.html')

@salaried_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        user_input = data.get('user_data')

        if not model_type or not user_input:
            return jsonify({'error': 'Missing model_type or user_data'}), 400

        numeric_fields = [
            'defaulter_neighbors', 'verified_neighbors', 'monthly_credit_bills',
            'bnpl_utilization_rate', 'mortgage_months_left', 'income-expense ratio',
            'owns_home', 'monthly_rent', 'recovery_days', 'sentiment_score'
        ]

        for field in numeric_fields:
            if field in user_input:
                user_input[field] = safe_float(user_input[field])

        if model_type == 'salaried':
            score = salaried_scorer.predict_alticred_score(user_input)
        else:
            return jsonify({'error': 'Invalid model type specified'}), 400

        return jsonify({'alticred_score': f"{score:.4f}"})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500
