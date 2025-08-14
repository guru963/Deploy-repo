import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import sys

# Add the models directory to the Python path
sys.path.insert(0, './models')
# Import your existing AltiCredScorer class
from models.alticred_salaried import AltiCredScorer

warnings.filterwarnings('ignore')

# --- 1. Load the Full Dataset ---
try:
    # Use a relative path to make it more portable
    full_df = pd.read_csv('salaried_dataset.csv')
    full_df.ffill(inplace=True)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Could not find 'salaried_dataset.csv'. Please make sure it's in the same directory.")
    exit()

# --- 2. Split Data into Training and Unseen Testing Sets ---
df_train, df_test = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['default_label'])
print(f"Data split into {len(df_train)} training rows and {len(df_test)} testing rows.")

# --- 3. Train the Scorer on the Training Data ONLY ---
temp_train_path = 'temp_train_data.csv'
df_train.to_csv(temp_train_path, index=False)

print("\n--- Training model on training data only... ---")
scorer = AltiCredScorer(file_path=temp_train_path)

# --- 4. Make Predictions on the Unseen Test Data ---
print("\n--- Evaluating model on unseen test data... ---")
y_true = []
y_pred = []

for index, row in df_test.iterrows():
    true_label = row['default_label']
    user_data = row.to_dict()
    predicted_score = scorer.predict_alticred_score(user_data)
    predicted_label = 1 if predicted_score > 0.5 else 0
    y_true.append(true_label)
    y_pred.append(predicted_label)

# --- 5. Report the Results ---
print("\n--- Model Evaluation Results ---")
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Overall Accuracy: {accuracy:.2%}\n")
print("✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Non-Defaulter', 'Defaulter']))
print("✅ Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# --- 6. NEW: Investigate Feature Importance ---
print("\n--- Investigating Feature Importance for Potential Data Leakage ---")

# Function to plot feature importance
def plot_importance(model, features, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=feature_importance.index)
        plt.title(f'Feature Importance - {title}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

# Plot for each base model that supports it
# plot_importance(scorer.trust_model, ['defaulter_neighbors', 'verified_neighbors', 'num_connections'], 'Digital Trust Model')
plot_importance(scorer.resilience_model, scorer.resilience_features, 'Resilience Model')
# The adaptability model can be a RF or LR, we check if it has the attribute
if hasattr(scorer.adapt_model, 'feature_importances_'):
    plot_importance(scorer.adapt_model, scorer.adapt_features, 'Adaptability Model')
else:
    print("\nAdaptability Model is Logistic Regression, cannot plot feature importance directly this way.")


# --- 7. Clean up the temporary file ---
os.remove(temp_train_path)