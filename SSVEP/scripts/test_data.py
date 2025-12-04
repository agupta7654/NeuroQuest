"""
Test script to read EEG data, normalize it, and print classification results
Similar to live_inference but with data output for debugging
"""

import time
import numpy as np
import argparse

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import tensorflow as tf

from preprocessing import SSVEPPreprocessor


class SSVEPDataTester:
    """Test EEG data with normalization and classification"""

    def __init__(self, model_path, board_id=BoardIds.CYTON_BOARD,
                 sampling_rate=250, window_size=1.0):
        """
        Initialize data tester

        Args:
            model_path: Path to trained model (.h5 file)
            board_id: BrainFlow board ID
            sampling_rate: Sampling rate in Hz
            window_size: Window size for classification in seconds
        """
        self.board_id = board_id
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.window_samples = int(sampling_rate * window_size)

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

        # Initialize preprocessor
        self.preprocessor = SSVEPPreprocessor(
            sampling_rate=sampling_rate,
            window_size=window_size
        )

        # BrainFlow setup
        self.params = BrainFlowInputParams()
        self.eeg_channels = None

        # Class labels
        self.class_labels = {0: '8Hz', 1: '14Hz', 2: 'Neither'}

        # Confidence threshold
        self.confidence_threshold = 0.6

    def setup_board(self, serial_port='COM3'):
        """Initialize and start the OpenBCI board"""
        # Set serial port if provided
        if serial_port:
            self.params.serial_port = serial_port
        
        # Create board instance with configured parameters
        self.board = BoardShim(self.board_id, self.params)
        
        print("Preparing OpenBCI board...")
        self.board.prepare_session()

        # Start streaming
        print("Starting data stream...")
        self.board.start_stream()
        print("Board ready! Waiting for data...")

        # Wait for initial data to accumulate
        time.sleep(2)

    def get_current_data_window(self):
        """
        Get the most recent window of EEG data

        Returns:
            data: Array of shape (n_channels, window_samples) at 250Hz
        """
        # OpenBCI native sampling rate is 250Hz
        native_sampling_rate = 250
        samples_needed = int(self.window_size * native_sampling_rate)

        # Get current data from board at native rate (250Hz)
        data = self.board.get_current_board_data(samples_needed)

        # Extract EEG channels in order: O1, O2, P3, P4, PO3, PO4
        eeg_data = data[[4, 5, 2, 3, 0, 1], :]

        # Check if we have enough data
        if eeg_data.shape[1] < samples_needed:
            return None

        # Return most recent window
        return eeg_data[:, -self.window_samples:]

    def classify_and_print(self, data_window):
        """
        Preprocess, classify a window of EEG data, and print results

        Args:
            data_window: Array of shape (n_channels, window_samples)

        Returns:
            prediction: Class label (0, 1, or 2)
            confidence: Confidence score
            raw_electrode_data: Raw data per electrode
            psd_unnormalized: PSD features before normalization
            psd_normalized: PSD features after normalization (0 to 1)
        """
        # Get raw electrode data
        electrode_names = ['O1', 'O2', 'P3', 'P4', 'PO3', 'PO4']
        electrode_data = []
        
        for i in range(data_window.shape[0]):
            electrode_data.append(data_window[i, :])
        
        # Preprocess using same method as SSVEP.ipynb
        # 1. Apply bandpass filter
        filtered = self.preprocessor.bandpass_filter(data_window)
        
        # 2. Extract PSD features using Welch's method (averaged across channels)
        _, psd_mean = self.preprocessor.extract_psd_features(filtered)
        
        # Store unnormalized PSD
        psd_unnorm = psd_mean.copy()
        
        # 3. Normalize to relative power (same as SSVEP.ipynb cell 9)
        # This converts to 0-1 range
        total_power = np.sum(psd_mean)
        if total_power > 0:
            psd_norm = psd_mean / total_power
        else:
            psd_norm = psd_mean

        # Reshape for model input: (1, n_features, 1)
        features = psd_norm.reshape(1, -1, 1)

        # Predict
        prediction_probs = self.model.predict(features, verbose=0)[0]

        # Get class and confidence
        prediction = np.argmax(prediction_probs)
        confidence = prediction_probs[prediction]
        class_name = self.class_labels[prediction]

        return prediction, confidence, class_name, electrode_names, electrode_data, psd_unnorm, psd_norm

    def run(self, duration=None, verbose=True):
        """
        Run test loop

        Args:
            duration: Duration in seconds (None for infinite)
            verbose: Print results
        """
        print("\n" + "=" * 80)
        print("Starting EEG Data Test")
        print("=" * 80)
        print("Reading normalized EEG data and classification results")
        print("Press Ctrl+C to stop")
        print("=" * 80 + "\n")

        start_time = time.time()
        iteration = 0

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Get data window
                data_window = self.get_current_data_window()

                if data_window is None:
                    time.sleep(0.1)
                    continue

                # Classify and get electrode data
                prediction, confidence, class_name, electrode_names, electrode_data, psd_unnorm, psd_norm = self.classify_and_print(data_window)

                # Determine command based on confidence
                if confidence >= self.confidence_threshold:
                    if prediction == 0:
                        command = "INPUT_8HZ"
                    elif prediction == 1:
                        command = "INPUT_14HZ"
                    else:
                        command = "INPUT_NEITHER"
                else:
                    command = "LOW_CONFIDENCE"

                # Print in the requested format
                if verbose:
                    # Print raw electrode data (first 10 samples)
                    for i, (name, raw_data) in enumerate(zip(electrode_names, electrode_data)):
                        raw_sample = ", ".join([f"{val:.2f}" for val in raw_data[:10]])
                        print(f"  Raw [{name}]: {raw_sample}...")
                    
                    print()  # Blank line for readability
                    
                    # Print unnormalized PSD features (all values)
                    unnorm_str = ", ".join([f"{val:.4f}" for val in psd_unnorm])
                    print(f"  Unnormalized PSD: [{unnorm_str}]")
                    
                    # Print normalized PSD features (all values, 0-1 range)
                    norm_str = ", ".join([f"{val:.6f}" for val in psd_norm])
                    print(f"  Normalized PSD:   [{norm_str}]")
                    
                    print(f"Command: {command}\n")

                iteration += 1

                # Small delay to control update rate
                time.sleep(0.25)

        except KeyboardInterrupt:
            print("\n\nStopping test...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Stop streaming and release resources"""
        print("Cleaning up...")
        if self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
        print("Cleanup complete!")


def main():
    """Main function for data testing"""
    parser = argparse.ArgumentParser(description='Test EEG data and classification')
    parser.add_argument('--model', type=str, default='SSVEP/neuroquest_model_s1.h5',
                       help='Path to trained model')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds (default: infinite)')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Confidence threshold (default: 0.6)')
    parser.add_argument('--board', type=str, default='cyton',
                       help='Board type: cyton, or synthetic')
    parser.add_argument('--serial-port', type=str, default='COM3',
                       help='Serial port for OpenBCI (e.g., /dev/ttyUSB0 or COM3)')

    args = parser.parse_args()

    # Map board name to BoardId
    board_map = {
        'cyton': BoardIds.CYTON_BOARD,
        'synthetic': BoardIds.SYNTHETIC_BOARD
    }

    if args.board not in board_map:
        print(f"Error: Unknown board type '{args.board}'")
        print(f"Available: {list(board_map.keys())}")
        return

    board_id = board_map[args.board]

    # Create tester
    tester = SSVEPDataTester(
        model_path=args.model,
        board_id=board_id,
        sampling_rate=250
    )

    # Set confidence threshold
    tester.confidence_threshold = args.confidence

    # Setup board with serial port
    tester.setup_board(serial_port=args.serial_port)

    # Run test
    tester.run(duration=args.duration)


if __name__ == '__main__':
    main()
