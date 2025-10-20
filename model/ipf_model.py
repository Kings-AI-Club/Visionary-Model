"""
TensorFlow Keras FNN for Homelessness Risk Prediction
=====================================================

Professionalized training script for IPF-generated synthetic data. This module
provides a clean, typed, and configurable pipeline to:
- load and validate data,
- preprocess features (age banding + one-hot),
- build either a logistic baseline or an FNN classifier,
- train with callbacks,
- calibrate a decision threshold on validation,
- evaluate and save artifacts (plots, metrics, model).

Data features expected:
- Gender (binary)
- Age (one-hot encoded: 7 age ranges from 0-17 to 65+)
- Drug, Mental, Indigenous, DV (binary risk factors)
- Location (one-hot encoded: ACT, NSW, NT, QLD, SA, TAS, VIC, WA)
- SHS_Client (binary indicator)
- Target: Homeless (binary)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Core ML
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Optional focal loss (if selected via config)
try:
    from focal_loss import BinaryFocalLoss  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    BinaryFocalLoss = None  # type: ignore

# Use a non-interactive backend so training doesn't block waiting for GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_paths: List[Path]
    model_type: str = os.environ.get("MODEL_TYPE", "fnn").lower()  # 'logistic' or 'fnn'
    learning_rate: float = float(os.environ.get("LR", "1e-3"))
    epochs: int = int(os.environ.get("EPOCHS", "40"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "256"))
    loss: str = os.environ.get("LOSS", "tversky").lower()  # 'tversky'|'bce'|'focal'
    figures_dir: Path = Path("model/figures")
    artifacts_dir: Path = Path("model/artifacts")
    model_path: Path = Path("homelessness_risk_model.h5")
    seed: int = int(os.environ.get("SEED", "42"))


def default_config() -> TrainConfig:
    candidates = [
        Path("model/data/synthetic_homelessness_data.csv"),
        Path("data/synthetic_homelessness_data.csv"),
        Path("synthetic-data/data/synthetic_homelessness_data.csv"),
    ]
    return TrainConfig(data_paths=candidates)


# --------------------------------------------------------------------------------------
# Data loading and preprocessing
# --------------------------------------------------------------------------------------
AGE_CATEGORIES: List[str] = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_data_path(candidates: Iterable[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find dataset. Tried: {', '.join(map(str, candidates))}. "
        "If you generated it elsewhere, set DATA_PATH env var or move the file."
    )


def load_data(cfg: TrainConfig) -> pd.DataFrame:
    env_path = os.environ.get("DATA_PATH")
    csv_path = Path(env_path) if env_path else resolve_data_path(cfg.data_paths)
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s samples, %s columns", f"{len(df):,}", df.shape[1])
    return df


def age_to_band(age: float) -> str:
    if pd.isna(age):
        return np.nan  # type: ignore[return-value]
    age = int(age)
    if age <= 17:
        return '0-17'
    if age <= 24:
        return '18-24'
    if age <= 34:
        return '25-34'
    if age <= 44:
        return '35-44'
    if age <= 54:
        return '45-54'
    if age <= 64:
        return '55-64'
    return '65+'


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    if 'Homeless' not in df.columns:
        raise ValueError("Expected 'Homeless' column in dataset")

    y = df['Homeless'].values
    logger.info(
        "Target prevalence: %.1f%% homeless (%s of %s)",
        y.mean() * 100,
        f"{y.sum():,}",
        f"{len(y):,}",
    )

    feature_cols = [c for c in df.columns if c != 'Homeless']
    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    if 'Age' not in X.columns:
        raise ValueError("Expected numeric 'Age' column before banding")

    X['Age_Category'] = X['Age'].apply(age_to_band)
    logger.info("Age bands: %s", X['Age_Category'].value_counts().to_dict())

    age_dummies = pd.get_dummies(X['Age_Category'], prefix='Age', dtype=int)
    age_dummies = age_dummies.reindex(
        columns=[f'Age_{c}' for c in AGE_CATEGORIES], fill_value=0
    )
    X = X.drop(['Age', 'Age_Category'], axis=1)
    X = pd.concat([X, age_dummies], axis=1)

    logger.info("Total features after encoding (%d)", len(X.columns))
    return X, y


def split_data(X: pd.DataFrame, y: np.ndarray, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    logger.info(
        "Split sizes -> train: %s, val: %s, test: %s",
        f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model(input_dim: int, cfg: TrainConfig) -> keras.Model:
    model_type = cfg.model_type
    if model_type == 'logistic':
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(1, activation='sigmoid')
        ])
        lr = float(os.environ.get('LR', cfg.learning_rate))
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
        lr = float(os.environ.get('LR', cfg.learning_rate))

    loss_name = cfg.loss
    if loss_name == 'bce':
        loss = keras.losses.BinaryCrossentropy()
    elif loss_name == 'focal':
        if BinaryFocalLoss is None:
            logger.warning("BinaryFocalLoss not available; falling back to BCE")
            loss = keras.losses.BinaryCrossentropy()
        else:
            loss = BinaryFocalLoss(gamma=2.0)
    else:  # default to tversky if available
        try:
            loss = keras.losses.Tversky(alpha=0.1, beta=0.9, name="tversky")
        except Exception:  # pragma: no cover
            logger.warning("Tversky loss unavailable; falling back to BCE")
            loss = keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


def calibrate_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    best_thr, best_acc = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        acc = ((y_proba > t).astype(int) == y_true).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr, best_acc


# --------------------------------------------------------------------------------------
# Training & Evaluation
# --------------------------------------------------------------------------------------
def train_and_evaluate(cfg: TrainConfig) -> Dict[str, float]:
    set_seeds(cfg.seed)

    df = load_data(cfg)
    X, y = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg.seed)

    X_train_array = X_train.values
    X_val_array = X_val.values
    X_test_array = X_test.values

    model = build_model(X_train_array.shape[1], cfg)
    logger.info("Model architecture:")
    model.summary(print_fn=lambda s: logger.info(s))


    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6
    )

    history = model.fit(
        X_train_array, y_train,
        validation_data=(X_val_array, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate
    train_loss, train_acc, train_auc = model.evaluate(X_train_array, y_train, verbose=0)
    val_loss, val_acc, val_auc = model.evaluate(X_val_array, y_val, verbose=0)
    test_loss, test_acc, test_auc = model.evaluate(X_test_array, y_test, verbose=0)

    y_val_proba = model.predict(X_val_array, verbose=0).flatten()
    best_thr, best_acc = calibrate_threshold(y_val, y_val_proba)
    logger.info("Calibrated threshold: %.3f (val acc=%.4f)", best_thr, best_acc)

    y_pred_proba = model.predict(X_test_array, verbose=0).flatten()
    y_pred = (y_pred_proba > best_thr).astype(int)

    logger.info("\n%s\n%s\n%s", "=" * 80, "CLASSIFICATION REPORT (Test)", "=" * 80)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=['Not Homeless', 'Homeless']))
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix:\n%s", cm)

    # Plot history
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cfg.figures_dir / 'figure_1_training_history.png'
    _plot_history(history, fig_path)
    logger.info("Saved training plot to: %s", fig_path)

    # Save model and metrics
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(cfg.model_path)
    logger.info("Saved model to: %s", cfg.model_path)

    metrics = {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "train_auc": float(train_auc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_auc": float(val_auc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_auc": float(test_auc),
        "threshold": float(best_thr),
        "threshold_val_acc": float(best_acc),
    }
    with open(cfg.artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(cfg.artifacts_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    return metrics


def _plot_history(history: keras.callbacks.History, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history.history['auc'], label='Train')
    axes[2].plot(history.history['val_auc'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Model AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    logger.info("%s", "=" * 80)
    logger.info("HOMELESSNESS RISK PREDICTION MODEL")
    logger.info("%s", "=" * 80)
    cfg = default_config()
    metrics = train_and_evaluate(cfg)
    logger.info("Training complete. Key metrics: %s", metrics)


if __name__ == "__main__":
    main()
