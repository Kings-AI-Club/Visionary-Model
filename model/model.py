# TensorFlow Keras FNN training for homelessness risk (simple baseline)
# - Loads data from data/synth.csv
# - Minimal preprocessing: one-hot encode categoricals, scale numerics
# - Simple FNN with Keras
# - Plots training vs validation accuracy with matplotlib.pyplot

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Core ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

# -----------------------------
# 1) Load data
# -----------------------------
csv_path = Path("data/synthetic_homelessness_data.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Couldn't find {csv_path}. Please confirm the path.")

df = pd.read_csv(csv_path)

# -----------------------------
# 2) Clean & type-cast
# -----------------------------
# Strip whitespace from column names just in case
df.columns = [c.strip() for c in df.columns]

# Expected schema (order doesn't matter as long as the names match)
expected_cols = [
    "gender",
    "indigenous",
    "mental_health",
    "drug_issue",
    "age",
    "location",
    "income_source",
    "education",
    "employment_status",
    "at_risk_homelessness",
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Normalize boolean/string booleans to actual bools
def to_bool_series(s):
    return s.map(
        {
            True: True, False: False,
            "True": True, "False": False,
            "true": True, "false": False,
            1: True, 0: False,
        }
    ).astype(bool)

bool_cols = ["indigenous", "mental_health", "drug_issue", "at_risk_homelessness"]
for c in bool_cols:
    if df[c].dtype == object or str(df[c].dtype).startswith(("int", "float")):
        df[c] = to_bool_series(df[c])

# Ensure age is numeric
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# -----------------------------
# 3) Split features/target
# -----------------------------
target_col = "at_risk_homelessness"
y = df[target_col].astype(int).values  # 0/1

feature_cols = [
    "gender", "indigenous", "mental_health", "drug_issue", "age",
    "location", "income_source", "education", "employment_status",
]
X = df[feature_cols].copy()

# -----------------------------
# 4) Preprocessing pipeline
# -----------------------------
categorical_cols = ["gender", "location", "income_source", "education", "employment_status"]
numeric_cols = ["age"]
binary_cols = ["indigenous", "mental_health", "drug_issue"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
        ("bin", "passthrough", binary_cols),
    ]
)

# Build a pipeline that ends with a Keras model; since Keras isn't scikit-compatible by default,
# we'll preprocess first, then feed NumPy arrays into Keras.
X_proc = preprocess.fit_transform(X)

input_dim = X_proc.shape[1]

# -----------------------------
# 5) Train/validation split
# -----------------------------
# With tiny datasets, stratify may fail; we guard for that.
stratify = y if len(np.unique(y)) > 1 and len(y) >= 6 else None
test_size = 0.25 if len(y) >= 12 else 0.5  # make sure we have a validation set even for tiny data

X_train, X_val, y_train, y_val = train_test_split(
    X_proc, y, test_size=test_size, random_state=42, stratify=stratify
)

# -----------------------------
# 6) Define a simple FNN
# -----------------------------
def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model(input_dim)

# -----------------------------
# 7) Train
# -----------------------------
# For tiny datasets, keep epochs small and use validation_data
epochs = 40 if len(y) > 100 else 60
batch_size = 16 if len(y) > 64 else max(1, min(8, len(X_train)))

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=0,
)

# -----------------------------
# 8) Plot train vs validation accuracy
# -----------------------------
plt.figure(figsize=(7,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 9) Quick evaluation printouts
# -----------------------------
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

print(f"Train size: {len(X_train)}  |  Val size: {len(X_val)}  |  Input dim: {input_dim}")
print(f"Final Train Accuracy: {train_acc:.3f}")
print(f"Final Val Accuracy:   {val_acc:.3f}")
