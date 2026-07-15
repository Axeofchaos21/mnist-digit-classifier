# MNIST Handwritten Digit Classification
Basic deep learning project implemented with PyTorch for 0-9 grayscale handwritten digit recognition.

## Project Overview
Built a Multi-Layer Perceptron (MLP) fully connected neural network to classify 28×28 MNIST digits. Implemented the complete end-to-end DL pipeline, including: dataset loading, tensor preprocessing, model training, performance evaluation, metric visualization, and trained weight saving.
- Standard PyTorch training & evaluation workflow
- GPU auto-detection for accelerated training
- Loss and accuracy curve visualization
- Persist trained model weights as .pth file

## Experiment Results
- Total Training Epochs: 5
- Final Test Accuracy: 97.68%
- Training Hardware: NVIDIA RTX 4060 Laptop GPU

## How to Run
1. Install required dependencies
```bash
pip install torch torchvision matplotlib
```

2. Execute MLP training script
```bash
python AC001.py
```
