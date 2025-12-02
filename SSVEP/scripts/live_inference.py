"""
Real-time SSVEP inference using OpenBCI and BrainFlow
Continuously reads EEG data, classifies SSVEP response, and sends commands via UDP
"""

import time
import socket
import numpy as np
from collections import deque
import argparse

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter
import tensorflow as tf

from preprocessing import SSVEPPreprocessor


class SSVEPRealtimeClassifier:
    """Real-time SSVEP classification from OpenBCI headset"""

    def __init__(self, model_path, board_id=BoardIds.CYTON_BOARD,
                 sampling_rate=125, window_size=1.0,
                 udp_ip='127.0.0.1', udp_port=5005):
        """
        Initialize real-time classifier

        Args:
            model_path: Path to trained model (.h5 file)
            board_id: BrainFlow board ID
            sampling_rate: Sampling rate in Hz (default: 125 Hz to match training data)
            window_size: Window size for classification in seconds
            udp_ip: UDP IP address for sending commands
            udp_port: UDP port for sending commands
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

        # UDP setup
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP socket configured: {udp_ip}:{udp_port}")

        # BrainFlow setup
        self.params = BrainFlowInputParams()
        self.board = BoardShim(board_id, self.params)
        self.eeg_channels = None

        # Data buffer
        self.data_buffer = None

        # Class labels
        self.class_labels = {0: '8Hz', 1: '14Hz'}

        # Confidence threshold
        self.confidence_threshold = 0.6

    def setup_board(self):
        """Initialize and start the OpenBCI board"""
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

        Note: OpenBCI samples at 250Hz, but we downsample to 125Hz to match training data

        Returns:
            data: Array of shape (n_channels, window_samples) at 125Hz
        """
        # OpenBCI native sampling rate is 250Hz
        # We need to get 2x samples to downsample to 125Hz
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

    def classify_window(self, data_window):
        """
        Classify a window of EEG data

        Args:
            data_window: Array of shape (n_channels, window_samples)

        Returns:
            prediction: Class label (0 or 1)
            confidence: Confidence score
            class_name: Human-readable class name
        """
        # Preprocess
        features = self.preprocessor.preprocess_epoch(data_window)

        # Reshape for model input: (1, n_features, 1)
        features = features.reshape(1, -1, 1)

        # Predict
        prediction_probs = self.model.predict(features, verbose=0)[0]

        # Get class and confidence
        prediction = np.argmax(prediction_probs)
        confidence = prediction_probs[prediction]
        class_name = self.class_labels[prediction]

        return prediction, confidence, class_name

    def send_udp_command(self, command):
        """
        Send command to Unity via UDP

        Args:
            command: String command to send
        """
        message = command.encode('utf-8')
        self.udp_socket.sendto(message, (self.udp_ip, self.udp_port))

    def run(self, duration=None, verbose=True):
        """
        Run real-time classification loop

        Args:
            duration: Duration in seconds (None for infinite)
            verbose: Print classification results
        """
        print("\n" + "=" * 60)
        print("Starting Real-Time SSVEP Classification")
        print("=" * 60)
        print("Class 0 (8Hz)  -> Command: INPUT_8HZ")
        print("Class 1 (14Hz) -> Command: INPUT_14HZ")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

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

                # Classify
                prediction, confidence, class_name = self.classify_window(data_window)

                # Only send command if confidence is above threshold
                if confidence >= self.confidence_threshold:
                    # Send UDP command
                    if prediction == 0:
                        self.send_udp_command("INPUT_8HZ")
                        command = "INPUT_8HZ"
                    else:
                        self.send_udp_command("INPUT_14HZ")
                        command = "INPUT_14HZ"

                    if verbose:
                        print(f"[{iteration:04d}] {class_name} | Confidence: {confidence:.3f} | Sent: {command}")
                else:
                    if verbose:
                        print(f"[{iteration:04d}] {class_name} | Confidence: {confidence:.3f} | LOW CONFIDENCE - No command sent")

                iteration += 1

                # Small delay to control update rate (adjust as needed)
                # For 1-second windows, you might update every 0.5 seconds for smoother response
                time.sleep(0.25)

        except KeyboardInterrupt:
            print("\n\nStopping classification...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Stop streaming and release resources"""
        print("Cleaning up...")
        if self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
        self.udp_socket.close()
        print("Cleanup complete!")


def main():
    """Main function for real-time inference"""
    parser = argparse.ArgumentParser(description='Real-time SSVEP classification')
    parser.add_argument('--model', type=str, default='neuroquest_champion_s1.h5',
                       help='Path to trained model')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds (default: infinite)')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Confidence threshold (default: 0.6)')
    parser.add_argument('--board', type=str, default='cyton_daisy',
                       help='Board type: cyton_daisy, cyton, or synthetic')
    parser.add_argument('--serial-port', type=str, default='COM3',
                       help='Serial port for OpenBCI (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--udp-ip', type=str, default='127.0.0.1',
                       help='UDP IP address')
    parser.add_argument('--udp-port', type=int, default=5005,
                       help='UDP port')

    args = parser.parse_args()

    # Map board name to BoardId
    board_map = {
        'cyton_daisy': BoardIds.CYTON_DAISY_BOARD,
        'cyton': BoardIds.CYTON_BOARD,
        'synthetic': BoardIds.SYNTHETIC_BOARD
    }

    if args.board not in board_map:
        print(f"Error: Unknown board type '{args.board}'")
        print(f"Available: {list(board_map.keys())}")
        return

    board_id = board_map[args.board]

    # Create classifier (125Hz to match training data)
    classifier = SSVEPRealtimeClassifier(
        model_path=args.model,
        board_id=board_id,
        sampling_rate=125,
        udp_ip=args.udp_ip,
        udp_port=args.udp_port
    )

    # Set serial port if provided
    if args.serial_port:
        classifier.params.serial_port = args.serial_port

    # Set confidence threshold
    classifier.confidence_threshold = args.confidence

    # Setup board
    classifier.setup_board()

    # Run classification
    classifier.run(duration=args.duration)


if __name__ == '__main__':
    main()
