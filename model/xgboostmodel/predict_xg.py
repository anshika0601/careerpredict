import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/realistic_career_dataset_22000_rows.csv")

# Features and labels
X = df[['education', 'interest', 'skill', 'personality']]
y = df['career']

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split for calibration (important)
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Base XGBoost model
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Calibrated model
calibrated_model = CalibratedClassifierCV(base_estimator=xgb, method='sigmoid', cv=3)
calibrated_model.fit(X_train, y_train)

#  Predict single career
def predict_career(education, interest, skill, personality):
    input_df = pd.DataFrame([{
        'education': education,
        'interest': interest,
        'skill': skill,
        'personality': personality,
    }])
    input_encoded = encoder.transform(input_df)
    pred_encoded = calibrated_model.predict(input_encoded)[0]
    career = label_encoder.inverse_transform([pred_encoded])[0]
    return career

#  Predict Top 3 career suggestions
def predict_top_3_careers(education, interest, skill, personality):
    input_df = pd.DataFrame([{
        'education': education,
        'interest': interest,
        'skill': skill,
        'personality': personality,
    }])
    input_encoded = encoder.transform(input_df)
    proba = calibrated_model.predict_proba(input_encoded)[0]
    top_3_idx = np.argsort(proba)[-3:][::-1]
    top_3_careers = label_encoder.inverse_transform(top_3_idx)
    top_3_probs = proba[top_3_idx]
    return list(zip(top_3_careers, top_3_probs.round(2)))

#  Example Test
print(predict_career("B.Tech", "Investigative", "Programming", "Analytical"))
print(predict_top_3_careers("B.Tech", "Investigative", "Programming", "Analytical"))

