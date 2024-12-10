# CAN Bus Intrusion Detection System

Deep learning-based intrusion detection system for automotive CAN bus networks using LSTM and attention mechanisms

## Features
- Bidirectional LSTM with attention for sequence processing
- Residual connections for improved gradient flow
- Focal loss for handling class imbalance
- Real-time CAN message sequence analysis

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset
Uses Car-Hacking Dataset containing:
- Normal CAN traffic
- DoS attacks
- Fuzzy attacks
- RPM spoofing
- Gear spoofing

## Model Architecture
- Input: Sequences of 5 CAN messages (10 features each)
- 3 Bidirectional LSTM layers (hidden dim: 128)
- Attention mechanism
- 3 Residual blocks (128→64→32)
- Binary classification output

## Performance
- Accuracy: 93.08%
- False Positive Rate: 15.3%
- False Negative Rate: 0.05%

## DataSet

You can download the dataset from here:
https://ocslab.hksecurity.net/Datasets/car-hacking-dataset


## Training
```bash
python3 CAN_Bus_Intrusion_Detection_System.py
```