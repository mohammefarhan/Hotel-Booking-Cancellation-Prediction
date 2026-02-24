# ====================================
# HOTEL CANCELLATION - FULL TRAINING
# FIXED VERSION
# ====================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -------------------------
# LOAD DATA
# -------------------------
print("Loading dataset...")
data = pd.read_csv("Hotel Reservations.csv")

# -------------------------
# FEATURE ENGINEERING
# -------------------------
print("Feature engineering...")

data = data.drop(columns=["Booking_ID"])

data["total_nights"] = (
    data["no_of_week_nights"] + data["no_of_weekend_nights"]
)

X = data.drop("booking_status", axis=1)
y = data["booking_status"]

# -------------------------
# FIX 1️⃣ SAVE LABEL ENCODER
# -------------------------
le = LabelEncoder()
y = le.fit_transform(y)

print("\nLabel Mapping:")
for i, label in enumerate(le.classes_):
    print(i, "=", label)

# Example output:
# 0 = Canceled
# 1 = Not_Canceled

# -------------------------
# ONE HOT ENCODING
# -------------------------
X = pd.get_dummies(X, drop_first=True)
model_columns = X.columns.tolist()

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# FIX 2️⃣ HANDLE IMBALANCE
# -------------------------
rf_params = {
    "n_estimators": [200],
    "max_depth": [10,12],
    "min_samples_split": [10],
    "min_samples_leaf": [4],
    "max_features": ["sqrt"],
    "class_weight": ["balanced"]   # IMPORTANT
}

print("\nRunning GridSearchCV...")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("\nBest Parameters:")
print(rf_grid.best_params_)

# -------------------------
# EVALUATION
# -------------------------
print("\nEvaluating model...")

train_pred = best_rf.predict(X_train)
test_pred = best_rf.predict(X_test)

print("\nTrain Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy :", accuracy_score(y_test, test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

# -------------------------
# SAVE EVERYTHING
# -------------------------
print("\nSaving model...")

joblib.dump(best_rf, "hotel_rf_model.pkl")
joblib.dump(model_columns, "model_columns.pkl")
joblib.dump(le, "label_encoder.pkl")   # VERY IMPORTANT

print("Model + Encoder saved successfully.")