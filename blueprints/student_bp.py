from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

student_bp = Blueprint('student', __name__, template_folder='templates')

class AdaptabilityScorer:
    def __init__(self, file_path='data/modified_student_data_v2.csv'):
        self.file_path = file_path
        self.df = self._load_and_clean_data()
        self._train_model()

    def _load_and_clean_data(self):
        df = pd.read_csv(self.file_path)
        loan_history_map = {'None': 0, '1x late': 1, '2x late': 1}
        df['is_defaulter'] = df['Education_Loan_History'].map(loan_history_map).fillna(0)
        df.ffill(inplace=True)
        return df

    def _train_model(self):
        numerical_features = [
            'Attendance_Rate (%)', 'Current_Academic_Standing (GPA)',
            'Stipend_Allowance_Amount (₹/month)', 'Internship_PartTime_Job_Income (₹/month)',
            'Ongoing_Certification_Courses', 'Achievement_Level', 'Median_Packages (LPA)'
        ]
        categorical_features = [
            'Location_Geotag (city/town/pincode)', 'Type_of_Institution',
            'Scholarships_Awards (Yes/No)', 'Student_Highlight'
        ]
        all_features = numerical_features + categorical_features
        df_adapt = self.df.dropna(subset=all_features + ['is_defaulter']).copy()

        X = df_adapt[all_features]
        y = df_adapt['is_defaulter']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            "Lasso (Logistic)": LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
            "ElasticNet (Logistic)": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(random_state=42)
        }

        best_auc = -np.inf
        for model in models.values():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
            if auc > best_auc:
                best_auc = auc
                self.adapt_model_pipeline = pipeline

    def predict(self, user_data):
        user_df = pd.DataFrame([user_data])
        score = self.adapt_model_pipeline.predict_proba(user_df)[0][0]
        return np.clip(score, 0, 1)

print("[Student] Loading and training model...")
scorer = AdaptabilityScorer()
print("[Student] Model ready!")

@student_bp.route('/', methods=['GET', 'POST'])
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
            score = scorer.predict(user_input)
        except Exception as e:
            score = f"Error: {e}"
    return render_template('index1.html', score=score)
