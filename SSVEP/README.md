# NeuroQuest - SSVEP BCI Gaming System

Real-time brain-computer interface using SSVEP (Steady-State Visual Evoked Potential) for gaming control.

## System Overview

This system classifies brain activity in response to flickering visual stimuli (8Hz and 14Hz) and sends control commands to a Unity game via UDP.

**Hardware:** OpenBCI Ultracortex Mark IV with Cyton + Daisy Board
**Channels:** O1, O2, P3, P4 (4 active electrodes)
**Sampling Rate:** 250 Hz

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Make sure you have:
- Python 3.8+
- TensorFlow 2.x
- BrainFlow
- OpenBCI hardware connected

### 2. Train Base Model

First, train a base model on the benchmark dataset:

```bash
cd scripts
python train_base.py
```

This will:
- Load data from `data/data_s1_64.mat`
- Extract 8Hz and 14Hz SSVEP responses
- Train a 1D CNN classifier
- Save model to `models/general_ssvep_model.h5`

**Note:** You may need to adjust the data indexing in `train_base.py` based on your actual MATLAB file structure. Check the shape and adjust the indexing accordingly.

### 3. Calibrate for Your Brain

Run a calibration session to fine-tune the model to your specific brain signals:

```bash
python calibration.py --serial-port /dev/ttyUSB0  # or COM3 on Windows
```

During calibration:
- You'll complete multiple trials focusing on 8Hz and 14Hz stimuli
- Each trial lasts 20 seconds (default)
- The script will automatically fine-tune the base model
- Calibrated model saved to `models/user_finetuned.h5`

**Calibration Options:**
```bash
python calibration.py \
    --trials 2 \              # Trials per frequency (default: 2)
    --trial-duration 20 \     # Duration per trial in seconds
    --board cyton_daisy \     # Board type
    --serial-port COM3 \      # Your serial port
    --save-data               # Save raw calibration data
```

### 4. Run Real-Time Classification

Start real-time SSVEP classification and send commands to Unity:

```bash
python live_inference.py --serial-port /dev/ttyUSB0
```

The system will:
- Stream EEG data from OpenBCI
- Classify every 1-second window
- Send UDP commands to Unity (127.0.0.1:5005):
  - `INPUT_8HZ` when detecting 8Hz focus
  - `INPUT_14HZ` when detecting 14Hz focus

**Live Inference Options:**
```bash
python live_inference.py \
    --model models/user_finetuned.h5 \  # Path to model
    --confidence 0.6 \                  # Confidence threshold
    --serial-port COM3 \                # Serial port
    --udp-ip 127.0.0.1 \               # Unity IP
    --udp-port 5005                     # Unity port
```

## Project Structure

```
neuroquest/
├── data/
│   ├── data_s1_64.mat                    # Training dataset
│   ├── Electrode_channels_information.csv # Channel mapping
│   └── calibration/                       # User calibration data
├── models/
│   ├── general_ssvep_model.h5            # Base model (30 subjects)
│   └── user_finetuned.h5                 # Calibrated model
├── scripts/
│   ├── preprocessing.py                   # Signal processing utilities
│   ├── ssvep_model.py                    # CNN model architecture
│   ├── train_base.py                     # Base model training
│   ├── calibration.py                    # User calibration
│   └── live_inference.py                 # Real-time classification
└── SSVEP.ipynb                           # Data exploration notebook
```

## System Architecture

### 1. Preprocessing Pipeline

- **Bandpass Filter:** 3-30 Hz (isolates SSVEP range)
- **Windowing:** 1-second epochs (250 samples)
- **Feature Extraction:** Power Spectral Density (Welch's method)
- **Channel Aggregation:** Average PSD across O1, O2, P3, P4

### 2. Model Architecture

1D CNN classifier:
- Input: PSD features (frequency bins, 1)
- 3x Conv1D blocks with BatchNorm and Dropout
- Dense layers with regularization
- Output: Softmax (2 classes: 8Hz, 14Hz)

### 3. Transfer Learning Strategy

1. **Base Model:** Trained on benchmark dataset (30 subjects)
2. **Fine-tuning:** Adapted to individual user (~40 seconds calibration)
3. **Lower learning rate** for fine-tuning preserves general features

### 4. Real-Time Pipeline

```
OpenBCI → BrainFlow → Preprocessing → Model → UDP → Unity
  250Hz     Buffer      PSD Feat.    CNN    Commands  Game
```

## Testing Without Hardware

Use synthetic board for development:

```bash
# Training
python train_base.py

# Calibration (synthetic data)
python calibration.py --board synthetic

# Live inference (synthetic data)
python live_inference.py --board synthetic
```

## Troubleshooting

### Issue: Model not loading
- Check that the model file exists in `models/`
- Ensure TensorFlow version compatibility

### Issue: BrainFlow connection error
- Verify serial port (use `ls /dev/tty*` on Linux/Mac or Device Manager on Windows)
- Check USB connection to OpenBCI
- Try synthetic board mode for testing

### Issue: Low classification accuracy
- Ensure electrodes are on O1, O2, P3, P4 positions
- Check electrode impedance (should be <50kΩ)
- Run longer calibration trials
- Ensure visual stimuli are clearly visible and stable

### Issue: Data indexing error in train_base.py
- The MATLAB file structure may vary
- Check the printed data shape
- Adjust indexing in `load_dataset_from_mat()` function
- Refer to dataset documentation for dimension order

## Unity Integration

In your Unity game, set up a UDP listener:

```csharp
// Unity C# UDP Receiver
UdpClient udp = new UdpClient(5005);
IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 5005);

void Update() {
    if (udp.Available > 0) {
        byte[] data = udp.Receive(ref remoteEP);
        string command = Encoding.UTF8.GetString(data);

        if (command == "INPUT_8HZ") {
            // Trigger game action for 8Hz
        } else if (command == "INPUT_14HZ") {
            // Trigger game action for 14Hz
        }
    }
}
```

## Performance Tuning

### Confidence Threshold
- **Higher (0.7-0.9):** Fewer false positives, slower response
- **Lower (0.5-0.6):** More responsive, more false positives
- Default: 0.6

### Window Size
- Current: 1.0 second
- Smaller windows: faster response, less accurate
- Larger windows: more accurate, slower response

### Update Rate
- Current: 0.25 second intervals
- Adjust `time.sleep()` in `live_inference.py`

## Dataset Information

**Source:** "An open dataset for human SSVEPs in BCI" (Scientific Data, 2024)
**Subjects:** 30
**Frequencies:** 40 (including 8Hz and 14Hz)
**Conditions:** Multiple depth stimuli

For this project:
- **Condition 1:** High-depth stimuli
- **Frequencies:** Index 7 (8Hz), Index 13 (14Hz)
- **Channels:** O1, O2, P3, P4

## License & Citation

If using this system for research, please cite the original SSVEP dataset paper.

## Support

For issues or questions:
1. Check troubleshooting section
2. Verify hardware connections
3. Test with synthetic board mode
4. Review BrainFlow documentation
