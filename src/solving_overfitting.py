import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping


# Path to JSON file containing MFCCs and genre labels
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data_10.json"


def load_data(data_path):
    """Load MFCC features and labels from JSON."""

    with open(data_path, "r") as fp:
        data = json.load(fp)

    mfccs = data["mfcc"]
    labels = data["labels"]

    # Find the longest MFCC sequence
    max_len = max(len(mfcc) for mfcc in mfccs)

    padded_mfccs = []

    for mfcc in mfccs:
        mfcc = np.array(mfcc)

        # Pad or trim MFCC sequences to the same length
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(
                mfcc,
                ((0, pad_width), (0, 0)),
                mode="constant"
            )
        else:
            mfcc = mfcc[:max_len, :]

        padded_mfccs.append(mfcc)

    X = np.array(padded_mfccs)
    y = np.array(labels)

    return X, y


def plot_history(history):
    """Plot training and validation accuracy/loss."""

    fig, axs = plt.subplots(2)

    # Accuracy
    axs[0].plot(
        history.history["accuracy"],
        label="training accuracy"
    )
    axs[0].plot(
        history.history["val_accuracy"],
        label="validation accuracy"
    )
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy")

    # Loss
    axs[1].plot(
        history.history["loss"],
        label="training loss"
    )
    axs[1].plot(
        history.history["val_loss"],
        label="validation loss"
    )
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Load MFCC data
    X, y = load_data(DATA_PATH)

    # Split into training (70%), validation (15%) and test (15%)
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

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    # Best-performing coursework model
    model = keras.Sequential([
        keras.layers.Flatten(
            input_shape=(X_train.shape[1], X_train.shape[2])
        ),

        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),

        keras.layers.Dense(10, activation="softmax")
    ])

    # Compile model
    optimiser = keras.optimizers.Adam(
        learning_rate=0.0001
    )

    model.compile(
        optimizer=optimiser,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Stop training when validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Final evaluation using previously unseen test data
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Plot training results
    plot_history(history)
