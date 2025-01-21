# Human Activity Recognition Using Deep Learning Models

This repository contains the implementation of various deep learning models for human activity recognition using video data. The project explores different approaches including image-based models, pose-based models, and a multi-stream model that combines both visual and pose information.

## Project Structure

```
.
├── Scratch-Trained Model for Image Data/
│   ├── main.ipynb              # Implementation of CNN-LSTM model for image data
│   └── requirements.txt        # Dependencies for image-based model
├── Scratch-Trained Model for Pose Data/
│   ├── main.ipynb              # Implementation of pose-based recognition model
│   └── requirements.txt        # Dependencies for pose-based model
├── Pre-Trained R3D-18 Model for Image Data/
│   ├── Pre-Trained R3D-18.ipynb # Implementation using pre-trained R3D-18
│   └── requirements.txt         # Dependencies for R3D-18 model
├── Pre-Trained MC3-18 Model for Image Data/
│   ├── Pre-Trained MC3-18.ipynb # Implementation using pre-trained MC3-18
│   └── requirements.txt         # Dependencies for MC3-18 model
├── Multi-Stream Model for both Image Data and Pose Data/
│   ├── main.ipynb              # Implementation of multi-stream fusion model
│   └── requirements.txt        # Dependencies for multi-stream model
└── README.md                   # Project documentation
```

## Models Overview

### 1. Scratch-Trained Model for Image Data
- A custom CNN-LSTM architecture trained from scratch
- Uses raw video frames as input
- Technologies:
  - TensorFlow/Keras
  - OpenCV
  - NumPy
  - Scikit-learn
  - Matplotlib/Seaborn for visualization

### 2. Scratch-Trained Model for Pose Data
- Focuses on human pose estimation and tracking
- Uses MediaPipe for pose extraction
- Technologies:
  - MediaPipe
  - TensorFlow/Keras
  - NumPy
  - Scikit-learn

### 3. Pre-Trained R3D-18 Model
- Utilizes the pre-trained R3D-18 architecture
- Transfer learning approach for video classification
- Technologies:
  - PyTorch
  - TorchVision
  - OpenCV
  - NumPy
  - YOLO for PPE detection

### 4. Pre-Trained MC3-18 Model
- Implements the MC3-18 architecture with pre-trained weights
- Similar to R3D-18 but with mixed 3D-2D convolutions
- Technologies:
  - PyTorch
  - TorchVision
  - OpenCV
  - NumPy

### 5. Multi-Stream Model
- Combines both visual and pose information
- Uses a fusion approach to merge different data streams
- Technologies:
  - TensorFlow/Keras
  - MediaPipe
  - OpenCV
  - Optuna for hyperparameter optimization
  - NumPy/Pandas

## Key Features

- Multiple model architectures for comparison
- Support for both image and pose-based analysis
- Real-time video processing capabilities
- Comprehensive evaluation metrics
- Frame-by-frame analysis
- PPE (Personal Protective Equipment) detection integration
- YouTube video download and processing support

## Dataset

The project uses the UCF50 dataset, which includes:
- 50 action categories
- Real-world videos from YouTube
- Varying video lengths and quality
- Diverse camera angles and backgrounds

## Implementation Details

### Data Processing
- Frame extraction and sequence generation
- Pose landmark detection and normalization
- Data augmentation techniques
- Preprocessing pipelines for both image and pose data

### Training
- Support for both scratch training and transfer learning
- Hyperparameter optimization using Optuna
- Early stopping and learning rate scheduling
- Class weight balancing for imbalanced data

### Evaluation
- Comprehensive metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- ROC curves and AUC scores
- Frame-by-frame analysis visualization

### Additional Features
- Real-time video processing
- PPE detection integration
- YouTube video download capability
- Visualization tools for analysis

## Requirements

Each model has its own `requirements.txt` file with specific dependencies. Common requirements include:
- Python 3.8+
- TensorFlow/PyTorch
- OpenCV
- MediaPipe
- NumPy/Pandas
- Scikit-learn
- Matplotlib/Seaborn
- YOLO
- Optuna

## Usage

1. Clone the repository
2. Install dependencies for the specific model you want to use
3. Download and prepare the UCF50 dataset
4. Run the respective notebook for training/evaluation

## Results

Each model includes comprehensive evaluation results:
- Training/validation curves
- Confusion matrices
- Classification reports
- ROC curves
- Frame-by-frame analysis graphs

## Future Work

- Integration of more pre-trained models
- Enhanced multi-stream fusion techniques
- Real-time deployment optimizations
- Additional activity classes support
- Mobile deployment considerations
