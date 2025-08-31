
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for Matplotlib

app = Flask(__name__, template_folder='templates') 

# --- Load only the pre-trained model ONCE on startup ---
try:
    with open('adaptability_model.pkl', 'rb') as f:
        MODEL_PIPELINE = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("\nERROR: Model file 'adaptability_model.pkl' not found.")
    print("Please run `python3 train_model.py` first to create the model file.\n")
    exit()

def create_feature_importance_plot():
    """Creates a feature importance plot image from the trained model."""
    model = MODEL_PIPELINE.named_steps['classifier']
    preprocessor = MODEL_PIPELINE.named_steps['preprocessor']
    
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    
    # Create a DataFrame for easy sorting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10) # Top 10 features
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='#3b82f6')
    ax.invert_yaxis() # Display top feature at the top
    ax.set_title('Top 10 Most Important Features', fontsize=16)
    ax.set_xlabel('Importance', fontsize=12)
    plt.tight_layout()
    
    # Save plot to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode buffer to a base64 string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

# Create the plot once on startup
FEATURE_PLOT_BASE64 = create_feature_importance_plot()

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
            
    return render_template('index1.html', score=score, feature_plot=FEATURE_PLOT_BASE64)

if __name__ == '__main__':
    app.run(debug=True)
