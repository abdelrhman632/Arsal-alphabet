import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv(
    r"C:\Users\Abdelrhman Yasser\Desktop\Uni\Term 8\NN\ArASL\arasl_landmarks_2.csv"
)

print("Dataset Shape:", df.shape)

# Features / Labels
X = df.drop("label", axis=1)
y = df["label"]

# SAME split used during training
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Load trained model
model = joblib.load(
    r"C:\Users\Abdelrhman Yasser\Desktop\Uni\Term 8\NN\ArASL\arasl_alphabet_model_2.pkl"
)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("\n========== MODEL EVALUATION ==========\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(y_test, y_pred))

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(y_test, y_pred)

print(cm)