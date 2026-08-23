from pathlib import Path

import librosa
import numpy as np


# -------------------------------------------------------
# Project settings
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "genres_original"
OUTPUT_PATH = PROJECT_ROOT / "data_10.npz"

SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# -------------------------------------------------------
# Audio augmentation
# -------------------------------------------------------

def add_white_noise(signal, noise_factor=0.005):
    """Add white noise to an audio signal."""

    noise = rng.standard_normal(len(signal)).astype(np.float32)
    return signal + noise_factor * noise


def pitch_shift(signal, sample_rate, n_steps=2):
    """Pitch-shift an audio signal."""

    return librosa.effects.pitch_shift(
        y=signal,
        sr=sample_rate,
        n_steps=n_steps
    )


# -------------------------------------------------------
# MFCC extraction
# -------------------------------------------------------

def extract_segments(
    signal,
    sample_rate,
    num_mfcc,
    n_fft,
    hop_length,
    num_segments
):
    """Extract MFCC features from fixed-length audio segments."""

    samples_per_segment = int(
        SAMPLES_PER_TRACK / num_segments
    )

    features = []

    for segment_index in range(num_segments):

        start = samples_per_segment * segment_index
        finish = start + samples_per_segment

        segment = signal[start:finish]

        if len(segment) < samples_per_segment:
            continue

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sample_rate,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )

        features.append(
            mfcc.T.astype(np.float32)
        )

    return features


# -------------------------------------------------------
# Dataset preparation
# -------------------------------------------------------

def prepare_dataset(
    dataset_path,
    output_path,
    num_mfcc=20,
    n_fft=1024,
    hop_length=256,
    num_segments=10
):
    """
    Prepare GTZAN data for training.

    Original tracks are split into:
        70% training
        15% validation
        15% testing

    Audio augmentation is applied ONLY to the training set.
    """

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    all_features = []
    all_labels = []
    all_splits = []
    all_track_ids = []
    all_variants = []

    mapping = []

    genre_folders = sorted(
        folder
        for folder in dataset_path.iterdir()
        if folder.is_dir()
    )

    print(f"Dataset found: {dataset_path}")
    print(f"Genres found: {len(genre_folders)}")

    for label_index, genre_folder in enumerate(genre_folders):

        genre_name = genre_folder.name
        mapping.append(genre_name)

        audio_files = sorted(
            file
            for file in genre_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".wav"
        )

        # Shuffle tracks deterministically
        indices = np.arange(len(audio_files))

        genre_rng = np.random.default_rng(
            RANDOM_SEED + label_index
        )

        genre_rng.shuffle(indices)

        audio_files = [
            audio_files[i]
            for i in indices
        ]

        # Split ORIGINAL tracks before augmentation
        total_tracks = len(audio_files)

        train_count = round(total_tracks * 0.70)
        validation_count = round(total_tracks * 0.15)

        train_files = audio_files[:train_count]

        validation_files = audio_files[
            train_count:
            train_count + validation_count
        ]

        test_files = audio_files[
            train_count + validation_count:
        ]

        track_groups = [
            ("train", train_files),
            ("validation", validation_files),
            ("test", test_files)
        ]

        print(f"\nGenre: {genre_name}")
        print(f"  Training tracks:   {len(train_files)}")
        print(f"  Validation tracks: {len(validation_files)}")
        print(f"  Test tracks:       {len(test_files)}")

        for split_name, files in track_groups:

            for file_number, file_path in enumerate(
                files,
                start=1
            ):

                if file_number == 1 or file_number % 10 == 0:
                    print(
                        f"  {split_name}: "
                        f"{file_number}/{len(files)}"
                    )

                try:
                    signal, sample_rate = librosa.load(
                        file_path,
                        sr=SAMPLE_RATE,
                        mono=True
                    )

                except Exception as error:
                    print(
                        f"  Skipping {file_path.name}: "
                        f"{error}"
                    )
                    continue

                # Make every track exactly 30 seconds
                if len(signal) < SAMPLES_PER_TRACK:

                    signal = np.pad(
                        signal,
                        (0, SAMPLES_PER_TRACK - len(signal))
                    )

                else:
                    signal = signal[:SAMPLES_PER_TRACK]

                # Training data gets augmentation.
                # Validation and test data remain untouched.
                if split_name == "train":

                    signal_versions = [
                        ("original", signal),
                        ("white_noise", add_white_noise(signal)),
                        (
                            "pitch_shift",
                            pitch_shift(
                                signal,
                                sample_rate
                            )
                        )
                    ]

                else:

                    signal_versions = [
                        ("original", signal)
                    ]

                for variant_name, audio_signal in signal_versions:

                    features = extract_segments(
                        audio_signal,
                        sample_rate,
                        num_mfcc,
                        n_fft,
                        hop_length,
                        num_segments
                    )

                    for mfcc in features:

                        all_features.append(mfcc)
                        all_labels.append(label_index)
                        all_splits.append(split_name)

                        all_track_ids.append(
                            f"{genre_name}/{file_path.stem}"
                        )

                        all_variants.append(
                            variant_name
                        )

    if not all_features:
        raise RuntimeError(
            "No MFCC features were extracted."
        )

    # ---------------------------------------------------
    # Ensure all MFCC arrays have equal dimensions
    # ---------------------------------------------------

    max_length = max(
        feature.shape[0]
        for feature in all_features
    )

    padded_features = []

    for feature in all_features:

        if feature.shape[0] < max_length:

            padding = max_length - feature.shape[0]

            feature = np.pad(
                feature,
                ((0, padding), (0, 0)),
                mode="constant"
            )

        elif feature.shape[0] > max_length:

            feature = feature[:max_length]

        padded_features.append(feature)

    X = np.stack(
        padded_features
    ).astype(np.float32)

    y = np.array(
        all_labels,
        dtype=np.int16
    )

    splits = np.array(all_splits)
    track_ids = np.array(all_track_ids)
    variants = np.array(all_variants)
    genre_mapping = np.array(mapping)

    # ---------------------------------------------------
    # Save compressed dataset
    # ---------------------------------------------------

    np.savez_compressed(
        output_path,
        mfcc=X,
        labels=y,
        splits=splits,
        track_ids=track_ids,
        variants=variants,
        mapping=genre_mapping
    )

    print("\nDataset preparation complete.")
    print(f"Saved to: {output_path}")
    print(f"MFCC shape: {X.shape}")

    print(
        "Training samples:",
        np.sum(splits == "train")
    )

    print(
        "Validation samples:",
        np.sum(splits == "validation")
    )

    print(
        "Test samples:",
        np.sum(splits == "test")
    )


# -------------------------------------------------------
# Best coursework feature configuration
# -------------------------------------------------------

if __name__ == "__main__":

    prepare_dataset(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        num_mfcc=20,
        n_fft=1024,
        hop_length=256,
        num_segments=10
    )