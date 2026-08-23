from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


# -------------------------------------------------------
# Project settings
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data_10.npz"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GRAPH_PATH = RESULTS_DIR / "improved_training_history.png"
METRICS_PATH = RESULTS_DIR / "improved_metrics.txt"

RANDOM_SEED = 42


# -------------------------------------------------------
# Load data
# -------------------------------------------------------

def load_data(data_path):
    """Load MFCC features, labels and dataset splits."""

    data = np.load(data_path)

    X = data["mfcc"]
    y = data["labels"]
    splits = data["splits"]
    genres = data["mapping"]

    return X, y, splits, genres


# -------------------------------------------------------
# Normalisation
# -------------------------------------------------------

def normalise_data(X_train, X_val, X_test):
    """
    Normalise MFCC values using statistics calculated
    only from the training set.
    """

    mean = np.mean(
        X_train,
        axis=(0, 1),
        keepdims=True
    )

    std = np.std(
        X_train,
        axis=(0, 1),
        keepdims=True
    )

    # Prevent division by zero
    std = np.where(std == 0, 1, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


# -------------------------------------------------------
# Model
# -------------------------------------------------------

def build_model(input_shape, number_of_genres):
    """Build a 1D CNN for music genre classification."""

    model = keras.Sequential([

        keras.layers.Input(
            shape=input_shape
        ),

        keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling1D(
            pool_size=2
        ),

        keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling1D(
            pool_size=2
        ),

        keras.layers.Conv1D(
            filters=256,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        keras.layers.BatchNormalization(),

        keras.layers.GlobalAveragePooling1D(),

        keras.layers.Dropout(0.4),

        keras.layers.Dense(
            128,
            activation="relu"
        ),

        keras.layers.Dropout(0.3),

        keras.layers.Dense(
            number_of_genres,
            activation="softmax"
        )
    ])

    return model


# -------------------------------------------------------
# Plot results
# -------------------------------------------------------

def plot_history(history, output_path):
    """Save and display training/validation graphs."""

    fig, axs = plt.subplots(
        2,
        figsize=(10, 8)
    )

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

    axs[0].set_title(
        "Training and Validation Accuracy"
    )

    axs[0].legend(
        loc="lower right"
    )

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

    axs[1].set_title(
        "Training and Validation Loss"
    )

    axs[1].legend(
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    print(
        f"Training graph saved to: {output_path}"
    )

    plt.show()


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":

    # Reproducibility
    np.random.seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)

    print("Loading MFCC data...")

    X, y, splits, genres = load_data(
        DATA_PATH
    )

    # ---------------------------------------------------
    # Use the track-level split created during preprocessing
    # ---------------------------------------------------

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[validation_mask]
    y_val = y[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Genres: {genres.tolist()}")

    print(
        f"\nTraining samples:   {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_val)}"
    )

    print(
        f"Test samples:       {len(X_test)}"
    )

    # ---------------------------------------------------
    # Normalise MFCC values
    # ---------------------------------------------------

    print("\nNormalising MFCC features...")

    X_train, X_val, X_test = normalise_data(
        X_train,
        X_val,
        X_test
    )

    # ---------------------------------------------------
    # Build model
    # ---------------------------------------------------

    model = build_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        ),
        number_of_genres=len(genres)
    )

    optimiser = keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimiser,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ---------------------------------------------------
    # Training controls
    # ---------------------------------------------------

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    reduce_learning_rate = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=100,
        batch_size=32,
        callbacks=[
            early_stopping,
            reduce_learning_rate
        ]
    )

    # ---------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    epochs_completed = len(
        history.history["loss"]
    )

    print("\nFinal Test Results")
    print("------------------")

    print(
        f"Test accuracy: {test_accuracy:.4f}"
    )

    print(
        f"Test loss:     {test_loss:.4f}"
    )

    print(
        f"Epochs completed: {epochs_completed}"
    )

    # ---------------------------------------------------
    # Save metrics
    # ---------------------------------------------------

    with open(
        METRICS_PATH,
        "w"
    ) as file:

        file.write(
            "Improved Music Genre Classification Model\n"
        )

        file.write(
            "==========================================\n\n"
        )

        file.write(
            f"Test accuracy: {test_accuracy:.4f}\n"
        )

        file.write(
            f"Test loss: {test_loss:.4f}\n"
        )

        file.write(
            f"Epochs completed: {epochs_completed}\n"
        )

        file.write(
            f"Training samples: {len(X_train)}\n"
        )

        file.write(
            f"Validation samples: {len(X_val)}\n"
        )

        file.write(
            f"Test samples: {len(X_test)}\n"
        )

    print(
        f"Metrics saved to: {METRICS_PATH}"
    )

    # ---------------------------------------------------
    # Save and display graph
    # ---------------------------------------------------

    plot_history(
        history,
        GRAPH_PATH
    )