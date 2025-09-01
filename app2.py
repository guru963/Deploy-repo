# app.py (Simplified Server, No SHAP)
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates') 

# Load only the pre-trained model ONCE on startup
try:
    with open('adaptability_model.pkl', 'rb') as f:
        MODEL_PIPELINE = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("\nERROR: Model file 'adaptability_model.pkl' not found.")
    print("Please run `python3 train_model.py` first to create the model file.\n")
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    if request.method == 'POST':
        try:
            user_input = {
                'Location_Geotag (city/town/pincode)': request.form['location'],
                'Type_of_Institution': request.form['institution_type'],
                'Attendance_Rate (%)': float(request.form['attendance']),
                'Scholarships_Awards (Yes/No)': request.form['scholarship'],
                'Current_Academic_Standing (GPA)': float(request.form['gpa']),
                'Stipend_Allowance_Amount (₹/month)': float(request.form['stipend']),
                'Internship_PartTime_Job_Income (₹/month)': float(request.form['internship_income']),
                'Ongoing_Certification_Courses': int(request.form['ongoing_courses']),
                'Achievement_Level': int(request.form['achievement']),
                'Student_Highlight': request.form['highlight'],
                'Median_Packages (LPA)': float(request.form['median_package'])
            }
            user_df = pd.DataFrame([user_input])
            score = MODEL_PIPELINE.predict_proba(user_df)[0][0]
        except Exception as e:
            score = f"Error: {e}"
            
    return render_template('index1.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)
