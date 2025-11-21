"""
Calibration Script for User-Specific Model Fine-Tuning
Records ~40 seconds of calibration data and fine-tunes the base model
"""

import time
import os
import numpy as np
from datetime import datetime
import argparse

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import SSVEPPreprocessor
from ssvep_model import fine_tune_model


class CalibrationRecorder:
    """Records calibration data for SSVEP model fine-tuning"""

    def __init__(self, board_id=BoardIds.CYTON_DAISY_BOARD, sampling_rate=125):
        """
        Initialize calibration recorder

        Args:
            board_id: BrainFlow board ID
            sampling_rate: Sampling rate in Hz (default: 125 Hz to match training data)
        """
        self.board_id = board_id
        self.sampling_rate = sampling_rate

        # BrainFlow setup
        self.params = BrainFlowInputParams()
        self.board = BoardShim(board_id, self.params)
        self.eeg_channels = None

        # Preprocessor
        self.preprocessor = SSVEPPreprocessor(sampling_rate=sampling_rate)

    def setup_board(self, serial_port=None):
        """Initialize and start the OpenBCI board"""
        if serial_port:
            self.params.serial_port = serial_port

        print("Preparing OpenBCI board...")
        self.board.prepare_session()

        # Get EEG channel indices
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        # Use first 4 channels (O1, O2, P3, P4)
        self.eeg_channels = self.eeg_channels[:4]
        print(f"Using channels: {self.eeg_channels}")

        # Start streaming
        print("Starting data stream...")
        self.board.start_stream()
        time.sleep(2)  # Wait for initial data
        print("Board ready!")

    def record_trial(self, duration, frequency_label):
        """
        Record a single trial

        Args:
            duration: Recording duration in seconds
            frequency_label: Label for this trial (0 for 8Hz, 1 for 14Hz)

        Returns:
            epochs: List of 1-second epochs
            labels: Corresponding labels
        """
        print(f"\nRecording {frequency_label}Hz trial for {duration} seconds...")
        print("Focus on the stimulus NOW!")

        # Clear buffer
        self.board.get_current_board_data(self.sampling_rate * 10)

        # Record (OpenBCI samples at 250Hz natively)
        time.sleep(duration)

        # Get recorded data at native rate (250Hz)
        native_sampling_rate = 250
        data = self.board.get_current_board_data(int(native_sampling_rate * duration))
        eeg_data = data[self.eeg_channels, :]

        # Downsample from 250Hz to 125Hz (take every 2nd sample)
        eeg_data = eeg_data[:, ::2]

        print(f"Recorded {eeg_data.shape[1]} samples")

        # Split into 1-second epochs
        epoch_samples = int(self.sampling_rate * 1.0)  # 1 second
        n_epochs = eeg_data.shape[1] // epoch_samples

        epochs = []
        labels = []

        for i in range(n_epochs):
            start = i * epoch_samples
            end = start + epoch_samples
            epoch = eeg_data[:, start:end]
            epochs.append(epoch)
            labels.append(0 if frequency_label == 8 else 1)

        print(f"Created {len(epochs)} epochs")
        return epochs, labels

    def run_calibration_protocol(self, trials_per_frequency=2, trial_duration=20):
        """
        Run complete calibration protocol

        Args:
            trials_per_frequency: Number of trials per frequency
            trial_duration: Duration of each trial in seconds

        Returns:
            all_epochs: Array of all recorded epochs
            all_labels: Corresponding labels
        """
        print("\n" + "=" * 60)
        print("SSVEP CALIBRATION PROTOCOL")
        print("=" * 60)
        print(f"You will complete {trials_per_frequency} trials for each frequency (8Hz and 14Hz)")
        print(f"Each trial lasts {trial_duration} seconds")
        print(f"Total calibration time: ~{(trials_per_frequency * 2 * trial_duration + trials_per_frequency * 2 * 5) // 60} minutes")
        print("=" * 60)

        all_epochs = []
        all_labels = []

        # Randomize trial order
        trials = ([8] * trials_per_frequency) + ([14] * trials_per_frequency)
        np.random.shuffle(trials)

        for i, freq in enumerate(trials):
            print(f"\n--- Trial {i+1}/{len(trials)} ---")
            print(f"Frequency: {freq}Hz")
            print("Prepare to focus on the stimulus...")
            print("Starting in: ", end='', flush=True)
            for countdown in range(3, 0, -1):
                print(f"{countdown}... ", end='', flush=True)
                time.sleep(1)
            print("GO!\n")

            # Record trial
            epochs, labels = self.record_trial(trial_duration, freq)
            all_epochs.extend(epochs)
            all_labels.extend(labels)

            # Rest between trials
            if i < len(trials) - 1:
                print(f"\nRest for 5 seconds...")
                time.sleep(5)

        print("\n" + "=" * 60)
        print("Calibration recording complete!")
        print(f"Total epochs: {len(all_epochs)}")
        print(f"8Hz epochs: {sum(1 for l in all_labels if l == 0)}")
        print(f"14Hz epochs: {sum(1 for l in all_labels if l == 1)}")
        print("=" * 60)

        return np.array(all_epochs), np.array(all_labels)

    def cleanup(self):
        """Stop streaming and release resources"""
        if self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()


def main():
    """Main calibration function"""
    parser = argparse.ArgumentParser(description='SSVEP Calibration')
    parser.add_argument('--base-model', type=str, default='neuroquest_champion_s1.h5',
                       help='Path to base model')
    parser.add_argument('--output-model', type=str, default='models/user_finetuned.h5',
                       help='Path to save fine-tuned model')
    parser.add_argument('--trials', type=int, default=2,
                       help='Trials per frequency (default: 2)')
    parser.add_argument('--trial-duration', type=int, default=20,
                       help='Trial duration in seconds (default: 20)')
    parser.add_argument('--board', type=str, default='cyton_daisy',
                       help='Board type: cyton_daisy, cyton, or synthetic')
    parser.add_argument('--serial-port', type=str, default='COM3',
                       help='Serial port for OpenBCI')
    parser.add_argument('--save-data', action='store_true',
                       help='Save calibration data to file')

    args = parser.parse_args()

    # Map board name to BoardId
    board_map = {
        'cyton_daisy': BoardIds.CYTON_DAISY_BOARD,
        'cyton': BoardIds.CYTON_BOARD,
        'synthetic': BoardIds.SYNTHETIC_BOARD
    }

    board_id = board_map.get(args.board, BoardIds.CYTON_DAISY_BOARD)

    # Create recorder
    recorder = CalibrationRecorder(board_id=board_id)

    try:
        # Setup board
        recorder.setup_board(args.serial_port if args.serial_port else None)

        # Run calibration
        epochs, labels = recorder.run_calibration_protocol(
            trials_per_frequency=args.trials,
            trial_duration=args.trial_duration
        )

        # Save raw data if requested
        if args.save_data:
            data_dir = 'data/calibration'
            os.makedirs(data_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            np.save(f'{data_dir}/epochs_{timestamp}.npy', epochs)
            np.save(f'{data_dir}/labels_{timestamp}.npy', labels)
            print(f"\nRaw data saved to {data_dir}/")

        # Preprocess data
        print("\nPreprocessing calibration data...")
        preprocessor = SSVEPPreprocessor()
        X, y = preprocessor.prepare_training_data(epochs, labels)
        print(f"Features shape: {X.shape}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Fine-tune model
        print("\nFine-tuning model on your data...")
        model, history = fine_tune_model(
            base_model_path=args.base_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_model_path=args.output_model,
            epochs=50,
            learning_rate=0.0001
        )

        # Evaluate
        print("\n" + "=" * 60)
        print("Calibration Complete!")
        print("=" * 60)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        print(f"Model saved to: {args.output_model}")
        print("\nYou can now run live_inference.py with your calibrated model!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")

    finally:
        recorder.cleanup()


if __name__ == '__main__':
    main()
