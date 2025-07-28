import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
data = pd.read_csv("winequality.csv")

# Handle missing values
data.dropna(inplace=True)  # Or use: data.fillna(data.mean(), inplace=True)

# Convert 'type' column to numeric if exists
if 'type' in data.columns:
    data['type'] = data['type'].map({'red': 0, 'white': 1})

# Convert quality into 3 categories
def map_quality(q):
    if q <= 4:
        return 0  # Bad
    elif q <= 6:
        return 1  # Neutral
    else:
        return 2  # Good

data['quality'] = data['quality'].apply(map_quality)

# Fill missing values with column mean
data.fillna(data.mean(), inplace=True)

y = data['quality']

# Balance using SMOTE
sm = SMOTE(random_state=42)
x_resampled, y_resampled = sm.fit_resample(x, y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(x_resampled, y_resampled)

# Save model
joblib.dump(model, "wine_quality_model.pkl")
print("✅ Model retrained and saved as wine_quality_model.pkl")
