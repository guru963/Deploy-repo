import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

class AdaptabilityScorer:
    # Corrected file path to local data folder
    def __init__(self, file_path='./data/modified_student_data_v2.csv'):
        self.file_path = file_path
        self.df = self._load_and_clean_data()
        self._train_model()

    def _load_and_clean_data(self):
        try:
            df = pd.read_csv(self.file_path)
            print("Adaptability Model: Dataset loaded.")
            loan_history_map = {'None': 0, '1x late': 1, '2x late': 1}
            df['is_defaulter'] = df['Education_Loan_History'].map(loan_history_map).fillna(0)
            df.ffill(inplace=True)
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            exit()

    def _train_model(self):
        # Define all features to be used
        numerical_features = [
            'Attendance_Rate (%)', 'Current_Academic_Standing (GPA)', 
            'Stipend_Allowance_Amount (₹/month)', 'Internship_PartTime_Job_Income (₹/month)',
            'Ongoing_Certification_Courses', 'Achievement_Level', 'Median_Packages (LPA)'
        ]
        categorical_features = [
            'Location_Geotag (city/town/pincode)', 'Type_of_Institution', 
            'Scholarships_Awards (Yes/No)', 'Student_Highlight'
        ]
        
        # Drop rows where the target or key features are missing
        all_features = numerical_features + categorical_features
        df_adapt = self.df.dropna(subset=all_features + ['is_defaulter']).copy()

        X = df_adapt[all_features]
        y = df_adapt['is_defaulter']

        # Create a preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Define models to test
        models = {
            "Lasso (Logistic)": LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
            "ElasticNet (Logistic)": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(random_state=42)
        }
        
        best_auc = -np.inf
        print("\n--- Adaptability Model Evaluation (using AUC) ---")
        for name, model in models.items():
            # Create a full pipeline with preprocessing and the model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model)])
            
            pipeline.fit(X_train, y_train)
            preds_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds_proba)
            acc = accuracy_score(y_test, pipeline.predict(X_test))
            print(f"{name}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                self.adapt_model_pipeline = pipeline
        
        print(f"\nSelected {type(self.adapt_model_pipeline.named_steps['classifier']).__name__} for Adaptability Score based on best AUC.")

    def predict(self, user_data):
        user_df = pd.DataFrame([user_data])
        # Probability of class 0 (not a defaulter)
        score = self.adapt_model_pipeline.predict_proba(user_df)[0][0]
        return np.clip(score, 0, 1)

if __name__ == '__main__':
    scorer = AdaptabilityScorer()
    print("\n--- Interactive Adaptability Score Calculator ---")
    while True:
        try:
            user_input = {
                'Location_Geotag (city/town/pincode)': input("Enter Location (Rural, Tier 1 (Urban), etc.): "),
                'Type_of_Institution': input("Enter Institution Type (Govt/Private): "),
                'Attendance_Rate (%)': float(input("Enter Attendance Rate (%): ")),
                'Scholarships_Awards (Yes/No)': input("Receiving Scholarship (Yes/No): "),
                'Current_Academic_Standing (GPA)': float(input("Enter GPA: ")),
                'Stipend_Allowance_Amount (₹/month)': float(input("Enter Stipend Amount: ")),
                'Internship_PartTime_Job_Income (₹/month)': float(input("Enter Internship Income: ")),
                'Ongoing_Certification_Courses': int(input("Enter Number of Ongoing Courses: ")),
                'Achievement_Level': int(input("Enter Achievement Level (1-5): ")),
                'Student_Highlight': input("Enter Student Highlight (e.g., Startup Founder, None): "),
                'Median_Packages (LPA)': float(input("Enter College Median Package (LPA): "))
            }
            score = scorer.predict(user_input)
            print(f"\nAdaptability Score: {score:.4f}\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")
        if input("Calculate for another student? (yes/no): ").lower() != 'yes':
            break
