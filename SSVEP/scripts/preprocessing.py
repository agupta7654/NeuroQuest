"""
SSVEP Preprocessing and Feature Extraction Utilities
Handles data preprocessing for both training and real-time inference
"""

import numpy as np
from scipy import signal
from scipy.signal import welch, butter, filtfilt


class SSVEPPreprocessor:
    """Preprocessing pipeline for SSVEP signals"""

    def __init__(self, sampling_rate=125, window_size=1.0, target_channels=None):
        """
        Initialize the preprocessor

        Args:
            sampling_rate: Sampling frequency in Hz (default: 125 Hz to match training data)
            window_size: Window size in seconds (default: 1.0 second)
            target_channels: List of channel indices to use (default: [0, 1, 2, 3] for O1, O2, P3, P4)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.n_samples = int(sampling_rate * window_size)
        self.target_channels = target_channels if target_channels is not None else [0, 1, 2, 3]

        # Design bandpass filter for SSVEP (3-30 Hz)
        self.lowcut = 3.0
        self.highcut = 30.0
        self.filter_order = 4
        self.b, self.a = self._design_bandpass_filter()

    def _design_bandpass_filter(self):
        """Design a Butterworth bandpass filter"""
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.filter_order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data):
        """
        Apply bandpass filter to EEG data

        Args:
            data: EEG data of shape (n_channels, n_samples) or (n_samples,)

        Returns:
            Filtered data with same shape as input
        """
        if data.ndim == 1:
            return filtfilt(self.b, self.a, data)
        else:
            return np.array([filtfilt(self.b, self.a, channel) for channel in data])

    def extract_psd_features(self, data, nperseg=None):
        """
        Extract Power Spectral Density features using Welch's method

        Args:
            data: EEG data of shape (n_channels, n_samples)
            nperseg: Length of each segment for Welch's method (default: n_samples)

        Returns:
            frequencies: Array of frequency bins
            psd_avg: Averaged PSD across channels
        """
        if nperseg is None:
            nperseg = self.n_samples

        # Ensure data is 2D (n_channels, n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels = data.shape[0]
        psd_channels = []

        for channel in range(n_channels):
            # Compute PSD using Welch's method
            frequencies, psd = welch(
                data[channel],
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling='density'
            )
            psd_channels.append(psd)

        # Average PSD across channels
        psd_avg = np.mean(psd_channels, axis=0)

        # Filter to frequency range of interest (3-30 Hz)
        freq_mask = (frequencies >= self.lowcut) & (frequencies <= self.highcut)
        frequencies = frequencies[freq_mask]
        psd_avg = psd_avg[freq_mask]

        return frequencies, psd_avg

    def preprocess_epoch(self, epoch_data):
        """
        Complete preprocessing pipeline for a single epoch

        Args:
            epoch_data: Raw EEG epoch of shape (n_channels, n_samples)

        Returns:
            psd_features: PSD features ready for model input
        """
        # Apply bandpass filter
        filtered = self.bandpass_filter(epoch_data)

        # Extract PSD features
        _, psd = self.extract_psd_features(filtered)

        return psd

    def prepare_training_data(self, epochs, labels):
        """
        Prepare multiple epochs for training

        Args:
            epochs: Array of shape (n_epochs, n_channels, n_samples)
            labels: Array of shape (n_epochs,)

        Returns:
            X: Feature matrix of shape (n_epochs, n_frequency_bins, 1)
            y: Labels array
        """
        n_epochs = epochs.shape[0]
        features = []

        for i in range(n_epochs):
            psd = self.preprocess_epoch(epochs[i])
            features.append(psd)

        X = np.array(features)
        # Reshape for Conv1D: (n_samples, n_features, 1)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, labels


def load_mat_data(mat_file, channel_indices, freq_indices=[7, 13], condition=1):
    """
    Load and extract relevant data from MATLAB file

    Args:
        mat_file: h5py file object
        channel_indices: List of channel indices to extract (e.g., [O1, O2, P3, P4])
        freq_indices: List of frequency indices (default: [7, 13] for 8Hz and 14Hz)
        condition: Condition index to extract (default: 1 for high-depth)

    Returns:
        epochs: Array of shape (n_epochs, n_channels, n_samples)
        labels: Array of labels (0 for 8Hz, 1 for 14Hz)
    """
    import h5py

    # Get the main data key
    data_key = list(mat_file.keys())[0]
    full_data = np.array(mat_file[data_key])

    # Expected shape: may vary, needs to be verified
    # Typical SSVEP datasets: (blocks, frequencies, channels, samples) or similar
    print(f"Original data shape: {full_data.shape}")

    # Extract data for specific condition, channels, and frequencies
    # This will need to be adjusted based on actual data structure
    epochs_list = []
    labels_list = []

    # Note: The exact indexing depends on the actual data structure
    # This is a template that needs to be adjusted
    for freq_idx, freq_val in enumerate(freq_indices):
        # Extract epochs for this frequency
        # Adjust indexing based on actual data structure
        # Example: data[condition, channels, :, frequency, blocks]
        pass

    return np.array(epochs_list), np.array(labels_list)


def extract_ssvep_from_mat(mat_file_path, channel_names=['O1', 'O2', 'P3', 'P4'],
                           freq_indices=[7, 13], condition=1):
    """
    Complete pipeline to extract SSVEP training data from MATLAB file

    Args:
        mat_file_path: Path to .mat file
        channel_names: Names of channels to extract
        freq_indices: Frequency indices (7=8Hz, 13=14Hz)
        condition: Condition index

    Returns:
        epochs: Preprocessed epochs
        labels: Corresponding labels
    """
    import h5py
    import pandas as pd

    # Load channel information
    channel_info = pd.read_csv('data/Electrode_channels_information.csv')

    # Get channel indices
    channel_indices = []
    for ch_name in channel_names:
        idx = channel_info[channel_info['Electrode'] == ch_name].index
        if len(idx) > 0:
            channel_indices.append(idx[0])

    # Load MATLAB file
    with h5py.File(mat_file_path, 'r') as mat_file:
        epochs, labels = load_mat_data(mat_file, channel_indices, freq_indices, condition)

    return epochs, labels
