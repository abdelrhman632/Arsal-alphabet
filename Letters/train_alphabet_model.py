import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load landmark dataset
df = pd.read_csv(r"C:\Users\Abdelrhman Yasser\Downloads\ArASL\arasl_landmarks_2.csv")

print("Dataset Loaded:", df.shape)

# Features / Labels
X = df.drop("label", axis=1)
y = df["label"]

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Validation accuracy
val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_pred))

# Test accuracy
test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print(classification_report(y_test, test_pred))

# Save model
joblib.dump(model, r"C:\Users\Abdelrhman Yasser\Downloads\ArASL\arasl_alphabet_model_2.pkl")

print("Model saved.")