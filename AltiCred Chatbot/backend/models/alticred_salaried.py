import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from scipy.sparse import hstack
import json
import re
import warnings

warnings.filterwarnings('ignore')

def safe_json_list_parser(x):
    try:
        s = str(x).strip()
        if not s.startswith('['):
            s = f"[{s}]"
        return json.loads(s.replace("'", "\""))
    except:
        return []

class AltiCredScorer:
    # Corrected file path to local data folder
    def __init__(self, file_path='./data/salaried_dataset.csv'):
        self.file_path = file_path
        self.df = self._load_and_clean_data()
        self._train_all_models()

    def _load_and_clean_data(self):
        try:
            df = pd.read_csv(self.file_path)
            print("Dataset loaded successfully.")
            df.ffill(inplace=True)
            return df
        except FileNotFoundError:
            print(f"Error: The file was not found at {self.file_path}")
            exit()

    def _train_all_models(self):
        print("\n--- Training All Base Models ---")
        self._train_digital_trust_model()
        self._train_resilience_model()
        self._train_adaptability_model()
        self._train_language_sentiment_model()
        self._train_meta_model()
        print("\n--- All Models Trained Successfully ---")

    def _train_digital_trust_model(self):
        df_trust = self.df.dropna(subset=['defaulter_neighbors', 'verified_neighbors', 'connections', 'default_label']).copy()
        df_trust['num_connections'] = df_trust['connections'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        
        features = ['defaulter_neighbors', 'verified_neighbors', 'num_connections']
        X = df_trust[features]
        y = df_trust['default_label']

        self.trust_scaler = StandardScaler().fit(X)
        X_scaled = self.trust_scaler.transform(X)
        
        input_dim = X_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoder_layer = Dense(2, activation='relu')(input_layer)
        self.trust_encoder = Model(inputs=input_layer, outputs=encoder_layer)
        
        X_encoded = self.trust_encoder.predict(X_scaled)
        self.trust_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.trust_model.fit(X_encoded, y)
        print("1. Digital Trust Model Trained.")

    def _train_resilience_model(self):
        # CORRECTED: Replaced the unstable Ridge regression with a robust RandomForestClassifier
        df_resilience = self.df.dropna(subset=['monthly_credit_bills', 'bnpl_utilization_rate', 'mortgage_months_left', 'income-expense ratio', 'upi_balances', 'emi_status_log', 'default_label']).copy()
        df_resilience['avg_upi_balance'] = df_resilience['upi_balances'].apply(lambda x: np.mean(safe_json_list_parser(x)) if safe_json_list_parser(x) else 0)
        df_resilience['missed_emi_count'] = df_resilience['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))

        self.resilience_features = ['monthly_credit_bills', 'bnpl_utilization_rate', 'mortgage_months_left', 'avg_upi_balance', 'income-expense ratio', 'missed_emi_count']
        X = df_resilience[self.resilience_features]
        y = df_resilience['default_label']

        self.resilience_scaler = StandardScaler().fit(X)
        X_scaled = self.resilience_scaler.transform(X)
        
        self.resilience_model = RandomForestClassifier(random_state=42)
        self.resilience_model.fit(X_scaled, y)
        print("2. Resilience Model Trained (Corrected).")

    def _train_adaptability_model(self):
        df_adapt = self.df.dropna(subset=['owns_home', 'monthly_rent', 'income-expense ratio', 'emi_status_log', 'recovery_days', 'monthly_credit_bills', 'mortgage_status', 'default_label']).copy()
        df_adapt['missed_emi_count'] = df_adapt['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))
        mortgage_dummies = pd.get_dummies(df_adapt['mortgage_status'], prefix='mortgage')
        self.adapt_mortgage_cols = mortgage_dummies.columns
        df_adapt = pd.concat([df_adapt, mortgage_dummies], axis=1)

        self.adapt_features = ['owns_home', 'monthly_rent', 'income-expense ratio', 'missed_emi_count', 'recovery_days', 'monthly_credit_bills'] + list(self.adapt_mortgage_cols)
        X = df_adapt[self.adapt_features]
        y = df_adapt['default_label']

        self.adapt_scaler = StandardScaler().fit(X)
        X_scaled = self.adapt_scaler.transform(X)
        
        rf = RandomForestClassifier(random_state=42).fit(X_scaled, y)
        lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_scaled, y)
        self.adapt_model = rf if accuracy_score(y, rf.predict(X_scaled)) > accuracy_score(y, lr.predict(X_scaled)) else lr
        print("3. Adaptability Model Trained.")

    def _train_language_sentiment_model(self):
        df_lang = self.df.dropna(subset=['user_posts', 'sentiment_score', 'default_label']).copy()
        df_lang['cleaned_posts'] = df_lang['user_posts'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        
        self.lang_vectorizer = TfidfVectorizer(max_features=100, stop_words='english').fit(df_lang['cleaned_posts'])
        X_text = self.lang_vectorizer.transform(df_lang['cleaned_posts'])
        X_sentiment = df_lang['sentiment_score'].values.reshape(-1, 1)
        X_combined = hstack([X_text, X_sentiment])
        y = df_lang['default_label']
        
        self.lang_model = LogisticRegression(random_state=42, max_iter=1000).fit(X_combined, y)
        print("4. Language Sentiment Model Trained.")

    def _train_meta_model(self):
        meta_features = pd.DataFrame(index=self.df.index)
        meta_features['trust_score'] = self._predict_digital_trust_score(self.df)
        meta_features['resilience_score'] = self._predict_resilience_score(self.df)
        meta_features['adaptability_score'] = self._predict_adaptability_score(self.df)
        meta_features['language_score'] = self._predict_language_sentiment_score(self.df)
        meta_features.fillna(meta_features.median(), inplace=True)
        
        self.meta_model = LogisticRegression(random_state=42).fit(meta_features, self.df['default_label'])
        print("5. Meta-Model Trained.")

    def _predict_digital_trust_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['num_connections'] = df['connections'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        X = df[['defaulter_neighbors', 'verified_neighbors', 'num_connections']]
        X_scaled = self.trust_scaler.transform(X)
        X_encoded = self.trust_encoder.predict(X_scaled)
        return self.trust_model.predict_proba(X_encoded)[:, 0]

    def _predict_resilience_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['avg_upi_balance'] = df['upi_balances'].apply(lambda x: np.mean(safe_json_list_parser(x)) if safe_json_list_parser(x) else 0)
        df['missed_emi_count'] = df['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))
        X = df[self.resilience_features]
        X_scaled = self.resilience_scaler.transform(X)
        return self.resilience_model.predict_proba(X_scaled)[:, 0]

    def _predict_adaptability_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['missed_emi_count'] = df['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))
        mortgage_dummies = pd.get_dummies(df['mortgage_status'], prefix='mortgage')
        df = pd.concat([df, mortgage_dummies], axis=1)
        for col in self.adapt_mortgage_cols:
            if col not in df.columns: df[col] = 0
        X = df[self.adapt_features]
        X_scaled = self.adapt_scaler.transform(X)
        return self.adapt_model.predict_proba(X_scaled)[:, 0]

    def _predict_language_sentiment_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['cleaned_posts'] = df['user_posts'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        X_text = self.lang_vectorizer.transform(df['cleaned_posts'])
        X_sentiment = df['sentiment_score'].fillna(0).values.reshape(-1, 1)
        X_combined = hstack([X_text, X_sentiment])
        return self.lang_model.predict_proba(X_combined)[:, 0]

    def predict_alticred_score(self, user_data):
        user_df = pd.DataFrame([user_data])
        
        s1 = self._predict_digital_trust_score(user_df)[0]
        s2 = self._predict_resilience_score(user_df)[0]
        s3 = self._predict_adaptability_score(user_df)[0]
        s4 = self._predict_language_sentiment_score(user_df)[0]
        
        base_scores = np.array([[s1, s2, s3, s4]])
        
        meta_prediction_proba = self.meta_model.predict_proba(base_scores)[0][0]
        
        risks = 1 - base_scores.flatten()
        total_risk = np.sum(risks)
        weights = risks / total_risk if total_risk > 0 else np.array([0.25, 0.25, 0.25, 0.25])
            
        weighted_score = np.sum(base_scores.flatten() * weights)
        
        final_score = (0.5 * meta_prediction_proba) + (0.5 * weighted_score)
        
        return np.clip(final_score, 0, 1)

if __name__ == '__main__':
    scorer = AltiCredScorer()

    print("\n--- AltiCred Integrated Score Calculator for Salaried Employees ---")
    while True:
        try:
            user_input = {
                'connections': input("Enter connections (e.g., user_1,user_2): "),
                'defaulter_neighbors': int(input("Enter number of defaulter neighbors: ")),
                'verified_neighbors': int(input("Enter number of verified neighbors: ")),
                'monthly_credit_bills': float(input("Enter Monthly Credit Bills: ")),
                'bnpl_utilization_rate': float(input("Enter BNPL Utilization Rate: ")),
                'mortgage_months_left': float(input("Enter Mortgage Months Left: ")),
                'upi_balances': input("Enter UPI balances (e.g., [1500, 2000] or 9500): "),
                'emi_status_log': input("Enter EMI status log (e.g., [1, 1, 0] or 1,1,1): "),
                'income-expense ratio': float(input("Enter Income-Expense Ratio: ")),
                'owns_home': int(input("Does the user own a home? (1 for yes, 0 for no): ")),
                'monthly_rent': float(input("Enter Monthly Rent (0 if home is owned): ")),
                'recovery_days': int(input("Enter typical recovery days: ")),
                'mortgage_status': input("Enter Mortgage Status (ongoing, paid, none): "),
                'user_posts': input("Enter a user post: "),
                'sentiment_score': float(input("Enter the post's sentiment score (-1 to 1): "))
            }

            final_score = scorer.predict_alticred_score(user_input)

            print("\n-----------------------------------------")
            print(f"AltiCred Score for the user: {final_score:.4f}")
            print("-----------------------------------------\n")

        except ValueError:
            print("\nInvalid input. Please ensure you enter numbers correctly.\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

        if input("Calculate for another user? (yes/no): ").lower() != 'yes':
            print("Exiting.")
            break
