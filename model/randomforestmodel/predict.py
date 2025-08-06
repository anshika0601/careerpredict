# Minimal example including model training and prediction in one script
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/realistic_career_dataset_22000_rows.csv")

X = df[['education', 'interest', 'skill', 'personality']]
y = df['career']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=42)
model.fit(X_encoded, y_encoded)

def predict_career(education, interest, skill, personality):
    input_df = pd.DataFrame([{
        'education': education,
        'interest': interest,
        'skill': skill,
        'personality': personality,
    }])
    input_encoded = encoder.transform(input_df)
    pred_encoded = model.predict(input_encoded)[0]
    career = label_encoder.inverse_transform([pred_encoded])[0]
    return career

def predict_top_3_careers(education, interest, skill, personality):
    input_df = pd.DataFrame([{
        'education': education,
        'interest': interest,
        'skill': skill,
        'personality': personality,
    }])
    input_encoded = encoder.transform(input_df)
    proba = model.predict_proba(input_encoded)[0]
    top_3_idx = np.argsort(proba)[-3:][::-1]
    top_3_careers = label_encoder.inverse_transform(top_3_idx)
    top_3_probs = proba[top_3_idx]
    return list(zip(top_3_careers, top_3_probs.round(2)))

# Example test
print(predict_career("B.Tech", "Investigative", "Programming", "Analytical"))
print(predict_top_3_careers("B.Tech", "Investigative", "Programming", "Analytical"))
