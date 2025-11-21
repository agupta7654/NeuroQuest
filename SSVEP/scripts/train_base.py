"""
Training script for base SSVEP model
Trains on benchmark dataset (30 subjects) to create a general model
"""

import os
import sys
import numpy as np
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from preprocessing import SSVEPPreprocessor
from ssvep_model import build_ssvep_cnn, train_model


def load_dataset_from_mat(mat_file_path, channel_info_path,
                          channel_names=['O1', 'O2', 'P3', 'P4'],
                          freq_8hz_idx=7, freq_14hz_idx=13, condition_idx=0):
    """
    Load and preprocess SSVEP data from MATLAB file

    Args:
        mat_file_path: Path to .mat file
        channel_info_path: Path to channel information CSV
        channel_names: List of channel names to use
        freq_8hz_idx: Index for 8Hz frequency
        freq_14hz_idx: Index for 14Hz frequency
        condition_idx: Condition index to use

    Returns:
        epochs: Array of shape (n_trials, n_channels, n_samples)
        labels: Array of labels (0 for 8Hz, 1 for 14Hz)
    """
    # Load channel information
    channel_info = pd.read_csv(channel_info_path)

    # Get channel indices
    channel_indices = []
    for ch_name in channel_names:
        idx = channel_info[channel_info['Electrode'] == ch_name].index
        if len(idx) > 0:
            channel_indices.append(idx[0])
        else:
            print(f"Warning: Channel {ch_name} not found in channel info")

    print(f"Using channels: {channel_names}")
    print(f"Channel indices: {channel_indices}")

    # Load MATLAB file
    with h5py.File(mat_file_path, 'r') as f:
        # Get data key (usually 'data' or similar)
        data_keys = [k for k in f.keys() if not k.startswith('#')]
        print(f"Available keys in MAT file: {data_keys}")

        if len(data_keys) == 0:
            raise ValueError("No data keys found in MAT file")

        data_key = data_keys[0]
        full_data = np.array(f[data_key])
        print(f"Full data shape: {full_data.shape}")
        print(f"Data structure: Check dimensions and adjust indexing accordingly")

        # NOTE: The exact indexing depends on your dataset structure
        # Common structure: (condition, channel, timepoint, frequency, block)
        # You may need to transpose or adjust based on actual structure

        epochs_list = []
        labels_list = []

        # Extract data for both frequencies
        # Assuming structure: (condition, channel, timepoint, frequency, block)
        # Adjust this based on your actual data structure!

        try:
            # For 8Hz trials (label 0)
            for block in range(full_data.shape[-1] if full_data.ndim == 5 else 1):
                if full_data.ndim == 5:
                    # (condition, channel, timepoint, frequency, block)
                    epoch_8hz = full_data[condition_idx, channel_indices, :, freq_8hz_idx, block]
                    epoch_14hz = full_data[condition_idx, channel_indices, :, freq_14hz_idx, block]
                else:
                    # Adjust for your actual data structure
                    print("Warning: Unexpected data dimensions. Please verify indexing.")
                    epoch_8hz = full_data[channel_indices, :]
                    epoch_14hz = full_data[channel_indices, :]

                epochs_list.append(epoch_8hz.T if epoch_8hz.shape[0] > epoch_8hz.shape[1] else epoch_8hz)
                labels_list.append(0)

                epochs_list.append(epoch_14hz.T if epoch_14hz.shape[0] > epoch_14hz.shape[1] else epoch_14hz)
                labels_list.append(1)

        except Exception as e:
            print(f"Error extracting data: {e}")
            print("Please check your data structure and adjust indexing in train_base.py")
            raise

    epochs = np.array(epochs_list)
    labels = np.array(labels_list)

    print(f"Extracted {len(epochs)} epochs")
    print(f"Epoch shape: {epochs[0].shape}")
    print(f"Label distribution: 8Hz={np.sum(labels==0)}, 14Hz={np.sum(labels==1)}")

    return epochs, labels


def main():
    """Main training pipeline"""

    # Configuration
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    CHANNEL_NAMES = ['O1', 'O2', 'P3', 'P4']
    SAMPLING_RATE = 250  # Hz
    WINDOW_SIZE = 1.0    # seconds

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("SSVEP Base Model Training")
    print("=" * 60)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    mat_file = os.path.join(DATA_DIR, 'data_s1_64.mat')
    channel_info = os.path.join(DATA_DIR, 'Electrode_channels_information.csv')

    epochs, labels = load_dataset_from_mat(
        mat_file, channel_info,
        channel_names=CHANNEL_NAMES
    )

    # Initialize preprocessor
    print("\n[2/5] Preprocessing data...")
    preprocessor = SSVEPPreprocessor(
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE
    )

    # Prepare training data
    X, y = preprocessor.prepare_training_data(epochs, labels)
    print(f"Feature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Split into train and validation
    print("\n[3/5] Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Build model
    print("\n[4/5] Building model...")
    input_shape = (X.shape[1], 1)  # (n_frequency_bins, 1)
    model = build_ssvep_cnn(input_shape, num_classes=2)

    print("\nModel Architecture:")
    model.summary()

    # Train model
    print("\n[5/5] Training model...")
    model_save_path = os.path.join(MODEL_DIR, 'general_ssvep_model.h5')

    history = train_model(
        model, X_train, y_train, X_val, y_val,
        model_save_path=model_save_path,
        epochs=100,
        batch_size=32
    )

    # Plot training history
    print("\nPlotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Model Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    print(f"Training history saved to {MODEL_DIR}/training_history.png")

    # Evaluate final model
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final Validation Accuracy: {final_acc*100:.2f}%")
    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Model saved to: {model_save_path}")


if __name__ == '__main__':
    main()
