import json
import os
import math
import librosa
import numpy as np
from pathlib import Path

def add_white_noise(signal, noise_factor=0.005):
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise


def pitch_shift(signal, sr, n_steps=2):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Data" / "genres_original"
JSON_PATH = PROJECT_ROOT / "data_10.json"

print("Dataset exists:", DATASET_PATH.exists())

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(
    dataset_path,
    json_path,
    num_mfcc=13,
    n_fft=2048,
    hop_length=512,
    num_segments=5
):
    """Extract MFCCs from dataset and save to JSON"""
    
    dataset_path = Path(dataset_path)
    json_path = Path(json_path)

    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

    # loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # skip root folder
        if Path(dirpath) == dataset_path:
            continue

        semantic_label = os.path.basename(dirpath)
        data["mapping"].append(semantic_label)
        print(f"\nProcessing: {semantic_label}")

        # process all audio files in genre folder
        for f in filenames:

            file_path = os.path.join(dirpath, f)

            try:
                signal, sample_rate = librosa.load(
                    file_path, sr=SAMPLE_RATE
                )
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue

            # original + augmentations
            signals = [
                signal,                               # original
                add_white_noise(signal),               # augmentation 1
                pitch_shift(signal, sample_rate)       # augmentation 2
            ]

            # process segments
            for augmented_signal in signals:
                for d in range(num_segments):

                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        y=augmented_signal[start:finish],
                        sr=sample_rate,
                        n_mfcc=num_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length
                    )

                    mfcc = mfcc.T

                    if mfcc.shape[1] > 0:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)

    # save MFCCs to JSON
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


# Run 3 configuration (BEST)

if __name__ == "__main__":
    save_mfcc(
        DATASET_PATH,
        JSON_PATH,
        num_mfcc=20,
        n_fft=1024,
        hop_length=256,
        num_segments=10
    )
