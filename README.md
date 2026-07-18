# 5G Signal Modulation Classifier

A deep learning model that classifies radio signal modulation types from raw IQ samples, trained with PyTorch and deployed on Hugging Face.

## Results

| Metric | Value |
|--------|-------|
| Dataset | 220,000 IQ signal samples, 11 modulation types, 20 SNR levels (-20dB to +18dB) |
| Accuracy (SNR >= 0dB) | ~80% (75.3% |
| Overall accuracy (all SNR) | ~60% (52.3%) |
| Model parameters | (1,093,835) |
| Training time | ~4 min on Google Colab T4 GPU |

## 3 Key Points

### 1. Problem and Data
Classified 11 radio modulation types (BPSK, QPSK, 8PSK, QAM16, QAM64, CPFSK, GFSK, PAM4, WB-FM, AM-SSB, AM-DSB) from raw IQ samples using the RadioML 2016.10a dataset. The dataset contains 220,000 synthetic signals generated with GNU Radio across 20 SNR levels from -20dB to +18dB.

### 2. Model and Results
Built a 1D CNN in PyTorch with two convolutional layers achieving **~80% accuracy at practical SNR levels (>= 0dB)** used in real 5G systems. Overall accuracy across all SNR levels is ~60%, reflecting the fundamental difficulty of classifying signals buried in noise at extreme low-SNR conditions.

### 3. Deployment
Published the trained model to [Hugging Face Hub](https://huggingface.co/tandrewu1/5G-Signal-Modulation-Classifier) using `PyTorchModelHubMixin`. Deployed a live [Gradio demo](https://huggingface.co/spaces/tandrewu1/5G-Signal-Modulation-Classifier) on Hugging Face Spaces for real-time browser-based inference.

## Architecture

- Input: 2 x 128 (I and Q channels, 128 time samples each)
- Conv1D(2, 64, kernel=8) + ReLU + MaxPool
- Conv1D(64, 128, kernel=5) + ReLU + MaxPool
- Fully connected (128*32 -> 256) + ReLU + Dropout(0.5)
- Output: 11 classes (softmax)

## Quick Start

```python
from model import ModulationClassifier

model = ModulationClassifier.from_pretrained("your-hf-username/5g-modulation-classifier")
model.eval()
```

## Files

- `train.ipynb` - Full training notebook (run on Google Colab)
- `app.py` - Gradio demo for Hugging Face Spaces
- `requirements.txt` - Dependencies for the Space
- `signal_examples.png` - Visualization of different modulation types
- `accuracy_vs_snr.png` - Accuracy vs SNR performance chart
- `confusion_matrix.png` - Confusion matrix at high SNR

## Dataset

[RadioML 2016.10a](https://zenodo.org/records/18397070) - CC BY-NC-SA 4.0 license.

## Links

- [Live Demo (Hugging Face Spaces)](https://huggingface.co/spaces/tandrewu1/5G-Signal-Modulation-Classifier)
- [Model (Hugging Face Hub)](https://huggingface.co/tandrewu1/5G-Signal-Modulation-Classifier)
