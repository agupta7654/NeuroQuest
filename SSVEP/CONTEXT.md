NEUROQUEST PROJECT CONTEXT & SPECIFICATIONS

1. PROJECT OVERVIEW

NeuroQuest is a BCI-controlled gaming system designed to treat neurocognitive disorders (ADHD) by enhancing focus. It uses a closed-loop system where brain signals directly control gameplay mechanics and game difficulty adapts to the user's neural focus level.

2. HARDWARE CONFIGURATION

Headset: OpenBCI Ultracortex Mark IV.

Board: Cyton + Daisy Board.

Sampling Rate: 250 Hz.

Active Electrodes (4 Channels):

O1, O2 (Primary Visual Cortex - Mandatory for SSVEP).

P3, P4 (Parietal - Supplementary).

Note: The code must handle input arrays expecting 4 channels, even if training data has 64.

Data Stream: Raw EEG data via BrainFlow library.

3. SOFTWARE ARCHITECTURE

The system follows a split-architecture design:

Backend (Python): Handles EEG streaming, signal processing, ML inference, and logic.

Frontend (Unity/C#): Handles game rendering (FocusFactory), visual stimuli (flickering), and user feedback.

Communication: UDP Networking (Python sends commands -> Unity receives).

IP: 127.0.0.1 (Localhost)

Port: 5005

4. MACHINE LEARNING PIPELINE

The system utilizes two parallel models.

A. Model 1: Input Control (SSVEP)

Function: Detects if user is focusing on 8Hz or 14Hz stimuli to trigger discrete game inputs.

Frequencies:

8 Hz (Target 1) -> Corresponds to Index 7 in training dataset.

14 Hz (Target 2) -> Corresponds to Index 13 in training dataset.

Training Strategy: Transfer Learning.

Base Model: Pre-trained on 30 subjects from the "Benchmark SSVEP Dataset" (Figshare).

Fine-Tuning: Re-trained on ~40 seconds of specific user calibration data.

Preprocessing:

Windowing: 1-second epochs (250 samples).

Feature Extraction: Power Spectral Density (PSD) via Welch's method.

Channel Aggregation: Average PSD across the 4 active channels (O1, O2, P3, P4).

Frequency Band: Interest range 3Hz - 30Hz.

Architecture: 1D CNN (Conv1D -> MaxPool -> Flatten -> Dense).

Input Shape: (FrequencyBins, 1)

Output: 2 classes (Softmax).

B. Model 2: Adaptive Difficulty (Focus Tracking)

Function: Measures overall cognitive load to adjust game speed/difficulty.

Metric: Higuchi Fractal Dimension (HFD).

Logic: Higher HFD = Higher Focus -> Increase Difficulty. Lower HFD = Distracted -> Decrease Difficulty.

Input: Time-series data from P3/P4 channels (Frontal/Parietal nodes are better for focus, but we work with what we have).

5. DATASET SPECIFICATIONS (For Training)

Source: "An open dataset for human SSVEPs" (Scientific Data, 2024).

Format: .mat files (e.g., data_s1_64.mat).

Structure: (Condition, Channel, TimePoint, Frequency, Block)

Filtering Rules for Training:

Condition: Use Index 1 (High-Depth stimuli).

Channels: Extract indices corresponding to ['O1', 'O2', 'P3', 'P4'] only.

Frequencies: Extract Index 7 (8Hz) and Index 13 (14Hz).

6. DIRECTORY STRUCTURE

/NeuroQuest_Root
│
├── /data                       # Local datasets (ignored by git)
│   ├── Electrode_channels_information.csv
│   ├── data_s1_64.mat
│   └── ...
│
├── /models                     # Saved .h5 models
│   ├── general_ssvep_model.h5  # Base model (30 subjects)
│   └── user_finetuned.h5       # Current user model
│
├── /scripts
│   ├── build_dataset.py        # Preprocesses .mat files into X/y npy arrays
│   ├── train_base.py           # Trains the general model
│   ├── calibration.py          # Records new user data & fine-tunes
│   └── live_inference.py       # Main loop: BrainFlow -> Model -> UDP
│
└── requirements.txt            # Python dependencies


7. KEY LIBRARIES

brainflow (Hardware interface)

mne (Signal filtering/Epoching)

scipy (Signal processing/Welch)

tensorflow (ML Models)

numpy & pandas (Data manipulation)

socket (UDP Communication)