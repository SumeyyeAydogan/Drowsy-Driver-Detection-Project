# 🚗 Drowsy Driver Detection Project

A deep learning project for detecting drowsy drivers using CNN (Convolutional Neural Network) with TensorFlow/Keras.

## 📋 Project Overview

This project implements a binary classification system to detect whether a driver is drowsy or not drowsy based on facial images. The system uses a custom CNN architecture optimized for binary classification.

## 🎯 Features

- **Binary Classification**: Drowsy (1) vs Not Drowsy (0)
- **Custom CNN Architecture**: 3 convolutional blocks with batch normalization and dropout
- **Data Augmentation**: Random flip, rotation, zoom for training
- **Run-based Organization**: Organized output structure with checkpoints and plots
- **Comprehensive Evaluation**: Confusion matrix, ROC curves, Precision-Recall curves
- **Automatic Plot Saving**: All visualizations saved as PNG files

## 🏗️ Architecture

### Model Structure
- **Input**: 224x224x3 RGB images
- **Output**: Binary classification (0 or 1)
- **Architecture**: 
  - 3 Conv blocks: [Conv → BN → ReLU] × 2 → MaxPool → Dropout
  - GlobalAveragePooling → Dense(128) → Dropout → Output(1)
- **Activation**: Sigmoid (binary classification)
- **Regularization**: L2 weight decay

### Data Pipeline
- **Image Size**: 224x224 pixels
- **Batch Size**: 32 (configurable)
- **Augmentation**: Horizontal flip, rotation, zoom
- **Normalization**: Rescaling to [0,1] range

## 📁 Project Structure

```
Drowsy-Driver-Detection-Project/
├── src/
│   ├── dataloader.py      # Data loading and preprocessing
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training functions
│   ├── evaluate.py        # Model evaluation
│   ├── utils.py           # Visualization and utilities
│   └── export.py          # Model saving
├── data/                  # Dataset (train/val/test)
├── runs/                  # Training outputs
│   └── run_001/
│       ├── models/        # Saved models
│       ├── plots/         # Training and evaluation plots
│       └── config.json    # Run configuration
├── main.py                # Main training script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install tensorflow scikit-learn seaborn matplotlib numpy pillow
```

### 2. Prepare Data
Organize your data in the following structure:
```
data/
├── train/                 # Training images (mixed drowsy/notdrowsy)
├── val/                   # Validation images (mixed drowsy/notdrowsy)
└── test/                  # Test images (mixed drowsy/notdrowsy)
```

**Important**: Image filenames must contain suffixes:
- `_1` for drowsy images (label: 1)
- `_0` for not drowsy images (label: 0)

Example: `001_glasses_nonsleepyCombination_1_notdrowsy_0.jpg`

### 3. Run Training
```bash
python main.py
```

This will:
- Create run directories
- Load and preprocess data
- Train the model (2 epochs by default)
- Save all plots and models
- Evaluate on test set

## 📊 Output Structure

After training, you'll find:

```
runs/run_001/
├── models/
│   └── final_model.h5    # Trained model
├── plots/
│   ├── training_history.png      # Accuracy and loss curves
│   ├── training_metrics.png      # Precision, recall, AUC curves
│   ├── confusion_matrix.png      # Confusion matrix
│   ├── roc_curve.png            # ROC curve
│   └── precision_recall_curve.png # PR curve
└── config.json                   # Run configuration
```

## ⚙️ Configuration

### Training Parameters
- **Epochs**: 2 (configurable in main.py)
- **Batch Size**: 32
- **Image Size**: 224x224
- **Learning Rate**: 1e-4 (Adam optimizer)
- **Loss Function**: Binary Crossentropy

### Model Parameters
- **Input Shape**: (224, 224, 3)
- **Weight Decay**: 1e-4 (L2 regularization)
- **Dropout**: 0.25 (conv layers), 0.5 (dense layers)

## 🔧 Customization

### Change Model Architecture
Edit `src/model.py` to modify the CNN architecture.

### Modify Training Parameters
Edit `main.py` to change epochs, batch size, etc.

### Add New Metrics
Edit `src/utils.py` to add new visualization functions.

## 📈 Performance Metrics

The system provides comprehensive evaluation:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification results

## 🐛 Troubleshooting

### Common Issues
1. **CUDA/GPU Issues**: Ensure TensorFlow GPU version is installed
2. **Memory Issues**: Reduce batch size in dataloader.py
3. **Import Errors**: Ensure virtual environment is activated

### Data Issues
- Check image file extensions (.jpg, .jpeg, .png)
- Verify filename suffixes (_1, _0)
- Ensure proper folder structure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- OpenCV for image processing capabilities
- The open-source community for various utilities and tools

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is designed for educational and research purposes. Always ensure proper safety measures when implementing driver monitoring systems in real vehicles.
