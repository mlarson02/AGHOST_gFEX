"""
A-GHOST: Cross-BX Anomaly Detection for gFEX
=============================================
Trains a Conv1D autoencoder on digitized gFEX MET data to detect anomalous
patterns across consecutive bunch crossings (BX-1, BX, BX+1). Reconstruction
error (MSE per event) is used as the anomaly score after training.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "mhxDigi", "mhyDigi",
    "msxDigi", "msyDigi",
    "metxDigi", "metyDigi",
    "metDigi", "sumEtDigi",
]

N_BX = 3
N_FEAT = len(BASE_FEATURES)
LATENT_DIM = 16


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df[BASE_FEATURES]


def build_cross_bx_windows(df: pd.DataFrame) -> np.ndarray:
    """
    For each event, concatenate features from BX-1, BX, and BX+1 into a
    single row, then reshape to (N, 3, 8). First and last rows are dropped
    because they lack a valid neighbour on one side.
    """
    df_prev = df.shift(1).add_suffix("_prev")
    df_next = df.shift(-1).add_suffix("_next")

    ordered_cols = (
        [f"{f}_prev" for f in BASE_FEATURES]
        + BASE_FEATURES
        + [f"{f}_next" for f in BASE_FEATURES]
    )

    df_context = pd.concat([df_prev, df, df_next], axis=1).iloc[1:-1].reset_index(drop=True)
    X = df_context[ordered_cols].to_numpy(dtype="float32")
    return X.reshape(-1, N_BX, N_FEAT)


def split_and_scale(X: np.ndarray, valid_size: float = 0.25, test_size: float = 0.25, random_state: int = 43):
    """
    50 / 25 / 25 train / valid / test split.
    valid_size and test_size are fractions of the full dataset.
    Scaler is fit only on training data.
    """
    X_train, X_temp = train_test_split(X, test_size=(valid_size + test_size), random_state=random_state)
    # split the held-out half evenly into valid and test
    X_valid, X_test = train_test_split(X_temp, test_size=0.5, random_state=random_state)

    scaler = StandardScaler()
    n_train, n_valid, n_test = X_train.shape[0], X_valid.shape[0], X_test.shape[0]

    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, N_FEAT)).reshape(n_train, N_BX, N_FEAT)
    X_valid_scaled = scaler.transform(X_valid.reshape(-1, N_FEAT)).reshape(n_valid, N_BX, N_FEAT)
    X_test_scaled  = scaler.transform(X_test.reshape(-1, N_FEAT)).reshape(n_test,  N_BX, N_FEAT)

    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


def save_splits(X_train, X_valid, X_test, out_dir: Path):
    """Save each scaled split as a flat (N, 3*8) parquet for future use."""
    cols = [f"{f}_{bx}" for bx in ("prev", "curr", "next") for f in BASE_FEATURES]
    for name, arr in [("train", X_train), ("valid", X_valid), ("test", X_test)]:
        path = out_dir / f"crossBX_{name}.parquet"
        pd.DataFrame(arr.reshape(arr.shape[0], -1), columns=cols).to_parquet(path, index=False)
        print(f"  Saved {name} split ({arr.shape[0]} events) → {path}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_autoencoder(n_bx: int = N_BX, n_feat: int = N_FEAT, latent_dim: int = LATENT_DIM):
    inputs = keras.Input(shape=(n_bx, n_feat))

    # Encoder
    x = layers.Conv1D(64,  kernel_size=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    # Decoder
    x = layers.Dense(n_bx * 128, activation="relu")(latent)
    x = layers.Reshape((n_bx, 128))(x)
    x = layers.Conv1D(64, kernel_size=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Conv1D(n_feat, kernel_size=2, padding="same", activation="linear")(x)

    autoencoder = keras.Model(inputs, outputs, name="conv_autoencoder")
    encoder     = keras.Model(inputs, latent,  name="encoder")

    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return autoencoder, encoder


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train(autoencoder, X_train, X_valid, epochs: int = 25, batch_size: int = 256):
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_valid, X_valid),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )
    return history


def save_onnx(model, out_path: str, n_bx: int = N_BX, n_feat: int = N_FEAT):
    import tf2onnx
    import onnx

    input_signature = [tf.TensorSpec(shape=(None, n_bx, n_feat), dtype=tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx.save(onnx_model, out_path)
    print(f"ONNX model saved to {out_path}")


def plot_loss(history, out_path: str = "loss_curve.png"):
    plt.figure()
    plt.plot(history.history["loss"],     label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Loss curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train A-GHOST cross-BX autoencoder")
    parser.add_argument("--input",      default="/workspace/samples/EnhancedBias_run00500306_gFEX_digitzedMET.parquet")
    parser.add_argument("--out-dir",    default="train_crossBX_gFEXMET_out",
                        help="Directory for all outputs (created if absent)")
    parser.add_argument("--epochs",     type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    print("Loading data...")
    df = load_data(args.input)

    print("Building cross-BX windows...")
    X = build_cross_bx_windows(df)
    print(f"  Dataset shape: {X.shape}")

    print("Splitting and scaling...")
    X_train, X_valid, X_test, scaler = split_and_scale(X)
    print(f"  Train: {X_train.shape}  Valid: {X_valid.shape}  Test: {X_test.shape}")

    print("Saving splits to parquet...")
    save_splits(X_train, X_valid, X_test, out_dir=out_dir)

    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    del X_test  # held out — not used during training

    print("Building model...")
    autoencoder, encoder = build_autoencoder(latent_dim=args.latent_dim)
    autoencoder.summary()

    plot_model(autoencoder, to_file=str(out_dir / "cross_BX_gFEXMET_autoencoder.png"),
               show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True)

    print("Training...")
    history = train(autoencoder, X_train, X_valid,
                    epochs=args.epochs, batch_size=args.batch_size)

    plot_loss(history, out_path=str(out_dir / "loss_curve.png"))

    model_path = out_dir / "autoencoder.keras"
    autoencoder.save(model_path)
    print(f"Model saved to {model_path}")

    print("Exporting to ONNX...")
    save_onnx(autoencoder, str(out_dir / "cross_BX_gFEXMET_autoencoder.onnx"))


if __name__ == "__main__":
    main()
