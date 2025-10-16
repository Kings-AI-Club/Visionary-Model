"""
TensorFlow Keras FNN for Homelessness Risk Prediction
======================================================
Trains on IPF-generated synthetic data with realistic demographic patterns.

Data features:
- Gender (binary)
- Age (one-hot encoded: 7 age ranges from 0-17 to 65+)
- Drug, Mental, Indigenous, DV (binary risk factors)
- Location (one-hot encoded: ACT, NSW, NT, QLD, SA, TAS, VIC, WA)
- SHS_Client (binary indicator)
- Target: Homeless (binary)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Core ML
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Use a non-interactive backend so training doesn't block waiting for GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("HOMELESSNESS RISK PREDICTION MODEL")
print("="*80)

# ============================================================================
# 1) Load IPF-generated data
# ============================================================================
print("\n1. Loading data...")
csv_path = Path("/Users/arona/Documents/GitHub/Visionary-Model/model/data/synthetic_homelessness_data.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Couldn't find {csv_path}. Please run ipf.py first.")

df = pd.read_csv(csv_path)
print(f"   Loaded {len(df):,} samples")
print(f"   Features: {df.shape[1]-1}")

# ============================================================================
# 2) Prepare features and target
# ============================================================================
print("\n2. Preparing features and target...")

# Target variable
y = df['Homeless'].values
print(f"   Target distribution: {y.sum():,} homeless ({y.mean()*100:.1f}%), {(~y.astype(bool)).sum():,} not homeless ({(1-y.mean())*100:.1f}%)")

# Features
feature_cols = [col for col in df.columns if col != 'Homeless']
X = df[feature_cols].copy()

# Convert all boolean columns to int (for TensorFlow compatibility)
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

# Reverse mapping: numeric age -> categorical age range
age_reverse_mapping = {
    10: '0-17',
    21: '18-24',
    30: '25-34',
    40: '35-44',
    50: '45-54',
    60: '55-64',
    70: '65+'
}

def age_to_band(age: float) -> str:
    """Bin raw age into categorical bands consistent with IPF brackets."""
    if pd.isna(age):
        return np.nan
    age = int(age)
    if age <= 17:
        return '0-17'
    elif age <= 24:
        return '18-24'
    elif age <= 34:
        return '25-34'
    elif age <= 44:
        return '35-44'
    elif age <= 54:
        return '45-54'
    elif age <= 64:
        return '55-64'
    else:
        return '65+'

# Convert raw numeric Age into the correct bands
X['Age_Category'] = X['Age'].apply(age_to_band)
print(f"   Age band distribution: {X['Age_Category'].value_counts().to_dict()}")

# One-hot encode Age with a fixed category order to ensure consistent columns
age_categories = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
age_dummies = pd.get_dummies(
    X['Age_Category'], prefix='Age', dtype=int
).reindex(columns=[f'Age_{c}' for c in age_categories], fill_value=0)
print(f"   Created {len(age_dummies.columns)} one-hot encoded age features: {list(age_dummies.columns)}")

# Drop original Age columns and add one-hot encoded age features
X = X.drop(['Age', 'Age_Category'], axis=1)
X = pd.concat([X, age_dummies], axis=1)

print(f"   Total features after encoding ({len(X.columns)}): {list(X.columns)}")

# ============================================================================
# 3) Train/validation/test split
# ============================================================================
print("\n3. Splitting data...")

# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")
print(f"   Test: {len(X_test):,} samples")

# ============================================================================
# 4) Feature scaling (optional - all features are now binary/one-hot encoded)
# ============================================================================
print("\n4. Preparing features for model...")

# No scaling needed - all features are binary or one-hot encoded
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

print(f"   All features are binary/one-hot encoded (no scaling required)")

# ============================================================================
# 5) Build FNN model
# ============================================================================
print("\n5. Building FNN model...")

def build_model(input_dim):
    """Build classifier. Default to logistic for higher accuracy on synth_weights."""
    model_type = os.getenv('MODEL_TYPE', 'logistic').lower()
    if model_type == 'logistic':
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(1, activation='sigmoid')
        ])
        lr = float(os.getenv('LR', '2e-3'))
    else:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        lr = float(os.getenv('LR', '1e-3'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model

input_dim = X_train_scaled.shape[1]
model = build_model(input_dim)

print(f"   Model architecture:")
model.summary()

# ============================================================================
# 6) Train model
# ============================================================================
print("\n6. Training model...")

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

# Convert to numpy arrays for TensorFlow
X_train_array = X_train_scaled.values
X_val_array = X_val_scaled.values
X_test_array = X_test_scaled.values

# Train
# Allow quick iteration control via env var
EPOCHS = int(os.getenv('EPOCHS', '100'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '256'))

history = model.fit(
    X_train_array, y_train,
    validation_data=(X_val_array, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================================================
# 7) Evaluate model
# ============================================================================
print("\n" + "="*80)
print("7. MODEL EVALUATION")
print("="*80)

# Training set
train_loss, train_acc, train_auc = model.evaluate(X_train_array, y_train, verbose=0)
print(f"\nTraining Set:")
print(f"   Loss: {train_loss:.4f}")
print(f"   Accuracy: {train_acc:.4f}")
print(f"   AUC: {train_auc:.4f}")

# Validation set
val_loss, val_acc, val_auc = model.evaluate(X_val_array, y_val, verbose=0)
print(f"\nValidation Set:")
print(f"   Loss: {val_loss:.4f}")
print(f"   Accuracy: {val_acc:.4f}")
print(f"   AUC: {val_auc:.4f}")

# Test set (final evaluation)
test_loss, test_acc, test_auc = model.evaluate(X_test_array, y_test, verbose=0)
print(f"\nTest Set:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_acc:.4f}")
print(f"   AUC: {test_auc:.4f}")

# Calibrate decision threshold on validation to maximize accuracy
y_val_proba = model.predict(X_val_array, verbose=0).flatten()
best_thr, best_acc = 0.5, 0.0
for t in np.linspace(0.05, 0.95, 181):
    acc = ((y_val_proba > t).astype(int) == y_val).mean()
    if acc > best_acc:
        best_acc, best_thr = acc, t
print(f"\nCalibrated threshold (validation): {best_thr:.3f} (val acc={best_acc:.4f})")

# Predictions on test set using calibrated threshold
y_pred_proba = model.predict(X_test_array, verbose=0).flatten()
y_pred = (y_pred_proba > best_thr).astype(int)

print(f"\n" + "="*80)
print("CLASSIFICATION REPORT (Test Set)")
print("="*80)
print(classification_report(y_test, y_pred, target_names=['Not Homeless', 'Homeless']))

print(f"\n" + "="*80)
print("CONFUSION MATRIX (Test Set)")
print("="*80)
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted")
print(f"              Not Homeless  Homeless")
print(f"Actual Not Homeless    {cm[0,0]:8d}    {cm[0,1]:8d}")
print(f"       Homeless        {cm[1,0]:8d}    {cm[1,1]:8d}")

# ============================================================================
# 8) Plot training history
# ============================================================================
print("\n8. Generating training plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AUC
axes[2].plot(history.history['auc'], label='Train')
axes[2].plot(history.history['val_auc'], label='Validation')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].set_title('Model AUC')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Save plot to a stable, accessible path without opening a window
fig_dir = Path('model/figures')
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / 'figure_1_training_history.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"   Saved training plot to: {fig_path}")

# Do not call plt.show() to avoid blocking interactive close

# ============================================================================
# 9) Save model
# ============================================================================
print("\n9. Saving model...")
model.save('homelessness_risk_model.h5')
print("   Model saved to: homelessness_risk_model.h5")
print("   Note: No scaler needed - all features are binary/one-hot encoded")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
