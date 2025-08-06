import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score

# Sample dataset (replace with your real dataset)
data = pd.read_csv("realistic_career_dataset_22000_rows.csv")

# Separate features and target
X = data[['education', 'interest', 'skill', 'personality']]
y = data['career']

# One-hot encode features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X)
X_encoded = encoder.transform(X)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Calibrated model for better probability output
calibrated_model = CalibratedClassifierCV(estimator=xgb_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# Predictions
y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)

# Top-3 accuracy
top3_acc = top_k_accuracy_score(y_test, y_proba, k=3)
print(f"Top-3 Accuracy: {top3_acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 🔮 Predict top 3 careers for new input
def predict_top_n(education, personality, skill, interest, top_n=3):
    # Create DataFrame with columns in the same order as training data
    input_df = pd.DataFrame({
        'education': [education],
        'interest': [interest],
        'skill': [skill],
        'personality': [personality]
    })

    # Use the original OneHotEncoder to transform the input
    input_encoded = encoder.transform(input_df)

    # Handle cases where all inputs are unknown and result in all zeros
    if np.all(input_encoded == 0):
         print("Prediction Error: All input features are unknown to the model.")
         return []

    # Use the calibrated model to get probabilities
    proba = calibrated_model.predict_proba(input_encoded)[0]

    top_n_indices = np.argsort(proba)[::-1][:top_n]
    top_labels = label_encoder.inverse_transform(top_n_indices)
    top_probs = proba[top_n_indices]

    print(" Top 3 Career Predictions:")
    for i, (label, prob) in enumerate(zip(top_labels, top_probs), 1):
        print(f"{i}. {label} ({prob:.2f})")


# Example
print("\n Example Top-3 Predictions:")
predict_top_n('B.Tech', 'Analytical', 'Programming', 'Investigative')