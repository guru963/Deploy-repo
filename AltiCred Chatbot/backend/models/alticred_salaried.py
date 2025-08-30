import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from scipy.sparse import hstack
import json
import re
import warnings

import shap
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


warnings.filterwarnings('ignore')

# --- Robust Helper Functions for User Input ---
def safe_json_list_parser(x):
    """
    Safely parses a string that might be a number, a comma-separated list,
    or a JSON list. Always returns a list.
    """
    try:
        s = str(x).strip()
        if not s.startswith('['):
            s = f"[{s}]"
        return json.loads(s.replace("'", "\""))
    except:
        return []

def display_shap_plots(explanation_object, current_score): # Added current_score for context
    """
    Helper function to display SHAP waterfall plot for a single instance
    with a more user-friendly explanation.
    """
    shap.initjs()
    print("\n--- Understanding Your AltiCred Score ---")
    print(f"Your current AltiCred Score is: {current_score:.2f}")

    if isinstance(explanation_object, shap.Explanation) and explanation_object.values.size > 0:
        print("\nLet's break down why your score is what it is:")
        print("---------------------------------------------")

        # Get feature names and their SHAP values
        feature_names = explanation_object.feature_names
        shap_values = explanation_object.values[0] # Assuming it's a single explanation
        feature_values = explanation_object.data[0] # Actual feature values

        # Create a list of impacts
        impacts = []
        for i, (name, shap_val, feat_val) in enumerate(zip(feature_names, shap_values, feature_values)):
            if shap_val > 0.05: # Significant positive impact (making score lower)
                impacts.append(f"🔴 Your **{name.replace('_', ' ').title()}** ({feat_val:.2f}) had a **strong negative impact** on your score.")
            elif shap_val < -0.05: # Significant negative impact (making score higher)
                impacts.append(f"🔵 Your **{name.replace('_', ' ').title()}** ({feat_val:.2f}) had a **positive impact** on your score.")
            elif shap_val != 0: # Small impact
                impacts.append(f"⚪ Your **{name.replace('_', ' ').title()}** ({feat_val:.2f}) had a small impact on your score.")
            # We can skip features with 0 impact if preferred

        if impacts:
            print("\nHere are the main factors influencing your AltiCred Score:")
            for impact in impacts:
                print(impact)
        else:
            print("No significant factors were identified as strongly influencing your score, or your score is neutral.")

        print("\n---------------------------------------------")
        print("\nVisual Explanation:")
        print("Below is a detailed chart. Red bars show factors that lowered your score, and blue bars show factors that raised it. The longer the bar, the bigger the impact.")
        
        # Plotting the waterfall
        plt.figure(figsize=(10, 6)) # Adjust figure size for better readability
        shap.plots.waterfall(explanation_object[0], max_display=len(feature_names))
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()
    else:
        print("Could not generate a detailed explanation. The prediction might be too simple, or there was an issue getting the insights.")


class AltiCredScorer:
    """
    An integrated scoring model that combines four different credit risk models
    using meta-stacking to produce a single, comprehensive AltiCred Score for Salaried Employees.
    This version includes SHAP explainability and an autoencoder for data generation.
    """
    def __init__(self, file_path='./data/salaried_dataset.csv'):
        self.file_path = file_path
        self.df = self._load_and_clean_data()
        
        # --- NEW: Initialize the analyzer once for efficiency ---
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        self.explainer = None  # Placeholder for the SHAP explainer
        self.meta_features_df = None # To store meta features for the explainer
        
        self._train_all_models()
        self._train_autoencoder()

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

        ratio = y.value_counts()[0] / y.value_counts()[1]
        self.trust_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=ratio)
        self.trust_model.fit(X_encoded, y)
        print("1. Digital Trust Model Trained.")

    def _train_resilience_model(self):
        df_resilience = self.df.dropna(subset=['monthly_credit_bills', 'bnpl_utilization_rate', 'mortgage_months_left', 'income-expense ratio', 'upi_balances', 'emi_status_log', 'default_label']).copy()
        df_resilience['avg_upi_balance'] = df_resilience['upi_balances'].apply(lambda x: np.mean(safe_json_list_parser(x)) if safe_json_list_parser(x) else 0)
        df_resilience['missed_emi_count'] = df_resilience['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))

        self.resilience_features = ['monthly_credit_bills', 'bnpl_utilization_rate', 'mortgage_months_left', 'avg_upi_balance', 'income-expense ratio', 'missed_emi_count']
        X = df_resilience[self.resilience_features]
        y = df_resilience['default_label']

        self.resilience_scaler = StandardScaler().fit(X)
        X_scaled = self.resilience_scaler.transform(X)
        
        self.resilience_model = RandomForestClassifier(random_state=42, class_weight='balanced')

        self.resilience_model.fit(X_scaled, y)
        print("2. Resilience Model Trained.")

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
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced').fit(X_scaled, y)
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced').fit(X_scaled, y)
        self.adapt_model = rf if accuracy_score(y, rf.predict(X_scaled)) > accuracy_score(y, lr.predict(X_scaled)) else lr
        print("3. Adaptability Model Trained.")

    def _train_language_sentiment_model(self):
        df_lang = self.df.dropna(subset=['user_posts', 'default_label']).copy()
        df_lang['cleaned_posts'] = df_lang['user_posts'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        
        get_score = lambda text: self.sentiment_analyzer.polarity_scores(text)['compound']
        df_lang['generated_sentiment_score'] = df_lang['user_posts'].apply(get_score)
        
        self.lang_vectorizer = TfidfVectorizer(max_features=100, stop_words='english').fit(df_lang['cleaned_posts'])
        X_text = self.lang_vectorizer.transform(df_lang['cleaned_posts'])
        
        X_sentiment = df_lang['generated_sentiment_score'].values.reshape(-1, 1)
        
        X_combined = hstack([X_text, X_sentiment])
        y = df_lang['default_label']
        
        self.lang_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced').fit(X_combined, y)
        print("4. Language Sentiment Model Trained (with Automated Sentiment).")

    def _train_meta_model(self):
        meta_features = pd.DataFrame(index=self.df.index)
        meta_features['trust_score'] = self._predict_digital_trust_score(self.df)
        meta_features['resilience_score'] = self._predict_resilience_score(self.df)
        meta_features['adaptability_score'] = self._predict_adaptability_score(self.df)
        meta_features['language_score'] = self._predict_language_sentiment_score(self.df)
        meta_features.fillna(meta_features.median(), inplace=True)
        
        self.meta_model = LogisticRegression(random_state=42, class_weight='balanced').fit(meta_features, self.df['default_label'])
        self.meta_features_df = meta_features
        self.explainer = shap.Explainer(self.meta_model, self.meta_features_df)
        print("5. Meta-Model Trained.")

    def _train_autoencoder(self):
        print("\n--- Training Autoencoder for Data Generation ---")
        autoencoder_features = self.df.select_dtypes(include=np.number).columns.tolist()
        df_ae = self.df[autoencoder_features].copy()
        df_ae.fillna(df_ae.median(), inplace=True)
        
        self.ae_scaler = MinMaxScaler().fit(df_ae)
        X_scaled = self.ae_scaler.transform(df_ae)
        
        input_dim = X_scaled.shape[1]
        encoding_dim = int(input_dim / 2) 

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        self.autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
        
        self.encoder = Model(input_layer, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        
        print("Autoencoder Trained.")

    def generate_synthetic_data(self, n_samples=100):
        """
        Generates new, synthetic data points using the trained autoencoder.
        """
        print(f"\n--- Generating {n_samples} Synthetic Data Points ---")
        latent_dim = self.encoder.input_shape[1]
        random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
        generated_data_scaled = self.decoder.predict(random_latent_vectors)
        generated_df = self.ae_scaler.inverse_transform(generated_data_scaled)
        generated_df = pd.DataFrame(generated_df, columns=self.df.select_dtypes(include=np.number).columns)
        print("Synthetic data generated.")
        return generated_df

    def _predict_digital_trust_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['num_connections'] = df['connections'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        X = df[['defaulter_neighbors', 'verified_neighbors', 'num_connections']]
        X_scaled = self.trust_scaler.transform(X)
        X_encoded = self.trust_encoder.predict(X_scaled)
        return self.trust_model.predict_proba(X_encoded)[:, 1]

    def _predict_resilience_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['avg_upi_balance'] = df['upi_balances'].apply(lambda x: np.mean(safe_json_list_parser(x)) if safe_json_list_parser(x) else 0)
        df['missed_emi_count'] = df['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))
        X = df[self.resilience_features]
        X_scaled = self.resilience_scaler.transform(X)
        return self.resilience_model.predict_proba(X_scaled)[:, 1]

    def _predict_adaptability_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['missed_emi_count'] = df['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))
        mortgage_dummies = pd.get_dummies(df['mortgage_status'], prefix='mortgage')
        df = pd.concat([df, mortgage_dummies], axis=1)
        for col in self.adapt_mortgage_cols:
            if col not in df.columns: df[col] = 0
        X = df[self.adapt_features]
        X_scaled = self.adapt_scaler.transform(X)
        return self.adapt_model.predict_proba(X_scaled)[:, 1]

    def _predict_language_sentiment_score(self, data):
        df = data.copy().reindex(columns=self.df.columns, fill_value=0)
        df['cleaned_posts'] = df['user_posts'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        
        get_score = lambda text: self.sentiment_analyzer.polarity_scores(text)['compound']
        df['generated_sentiment_score'] = df['user_posts'].apply(get_score)
        
        X_text = self.lang_vectorizer.transform(df['cleaned_posts'])
        X_sentiment = df['generated_sentiment_score'].values.reshape(-1, 1)
        
        X_combined = hstack([X_text, X_sentiment])
        return self.lang_model.predict_proba(X_combined)[:, 1]

    def predict_alticred_score(self, user_data):
        user_df = pd.DataFrame([user_data])
        
        s1 = self._predict_digital_trust_score(user_df)[0]
        s2 = self._predict_resilience_score(user_df)[0]
        s3 = self._predict_adaptability_score(user_df)[0]
        s4 = self._predict_language_sentiment_score(user_df)[0]
        
        base_scores = np.array([[s1, s2, s3, s4]])
        
        meta_predict_df = pd.DataFrame(base_scores, columns=['trust_score', 'resilience_score', 'adaptability_score', 'language_score'])
        
        shap_explanation = self.explainer(meta_predict_df)

        meta_prediction_proba = self.meta_model.predict_proba(base_scores)[0][1]
        
        risks = base_scores.flatten()
        total_risk = np.sum(risks)
        
        weights = risks / total_risk if total_risk > 0 else np.array([0.25, 0.25, 0.25, 0.25])
        
        weighted_score = np.sum(risks * weights)
        
        risk_score = (0.5 * meta_prediction_proba) + (0.5 * weighted_score)
        final_score = 1 - risk_score
        return np.clip(final_score, 0, 1), meta_predict_df, shap_explanation

    def evaluate_model(self):
        """
        Calculates and prints performance metrics for the meta-model.
        """
        print("\n--- Evaluating Meta-Model Performance ---")
        
        # Make predictions on the training data's meta-features
        y_true = self.df['default_label']
        y_pred = self.meta_model.predict(self.meta_features_df)
        y_proba = self.meta_model.predict_proba(self.meta_features_df)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        rmse = np.sqrt(mean_squared_error(y_true, y_proba))
        r2 = r2_score(y_true, y_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print("-----------------------------------------")


if __name__ == '__main__':
    scorer = AltiCredScorer()
    
    # Evaluate the model's performance on the training data after training
    scorer.evaluate_model()
    
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
                'user_posts': input("Enter a user post (e.g., 'I had a great week'): ")
            }

            final_score, _, shap_explanation = scorer.predict_alticred_score(user_input)
            
            # Use the new helper function to handle the plotting logic
            display_shap_plots(shap_explanation, final_score)

            print("\n-----------------------------------------")
            print(f"AltiCred Score for the user: {final_score:.4f}")
            print("-----------------------------------------\n")

        except ValueError as ve:
            print(f"\nInvalid input: {ve}. Please ensure you enter numbers correctly.\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

        if input("Calculate for another user? (yes/no): ").lower() != 'yes':
            print("Exiting.")
            break
