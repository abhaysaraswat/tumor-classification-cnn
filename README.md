# Brain Tumor Detector

This project builds a **Convolutional Neural Network (CNN)** in **TensorFlow & Keras** to classify brain MRI images as having a tumor or not. The model is trained on an augmented dataset of MRI brain images, achieving **98.63% accuracy on validation** and **98.47% accuracy on the test set**. The dataset and complete code are available in the repository.

## Table of Contents
- [About the Data](#about-the-data)
- [Data Augmentation](#data-augmentation)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [License](#license)

---

## About the Data

The dataset used for training consists of MRI brain images and is publicly available on Kaggle. It contains **two folders**:
- **yes/**: 155 images labeled as tumor.
- **no/**: 98 images labeled as notumor.

You can download the dataset from [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The original dataset contained **7022 images**

---

## Data Augmentation

The original dataset contained **four types of tumors** (Glioma, Meningioma, Pituitary, and others), but to simplify the classification task, we **merged all tumor types into a single class**: **"tumor"** vs. **"non-tumor"**. After merging, the data contained:

- **Tumor class**: 906 images.
- **Non-tumor class**: 405 images.


## Data Preprocessing

Each image goes through the following preprocessing steps:
1. **Normalization**: Pixel values are scaled between 0 and 1 to improve training performance.


---

## Neural Network Architecture

The CNN model architecture is as follows:
- **Conv2D layers**: Three convolutional layers with ReLU activation.
- **MaxPooling2D layers**: Pooling layers after each convolutional block to downsample.
- **Dropout layers**: Applied after every pooling layer to avoid overfitting.
- **Flatten layer**: Converts the 3D feature maps into 1D vectors.
- **Dense layers**: Fully connected layer with 128 units.
- **Output layer**: A single unit with a sigmoid activation function (since it's a binary classification task).


---

## Training the Model

The model was trained for **15 epochs** using an **Adam optimizer** with a learning rate of **0.001**. Below are the training statistics:

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------|---------------------|-----------------|
| 1     | 76.36%            | 0.5398        | 88.63%              | 0.2596          |
| 5     | 96.65%            | 0.0868        | 93.75%              | 0.1299          |
| 10    | 98.01%            | 0.0476        | 97.71%              | 0.0515          |
| 15    | 99.16%            | 0.0177        | 98.63%              | 0.0356          |

---

## Results

The best model achieved **98.63% validation accuracy** and **98.47% test accuracy**. These are the metrics:
- **Test Loss**: 0.0284
- **Test Accuracy**: 98.47%
- **Confusion Matrix**: Available in the repository.

---

## How to Run

To run this project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abhaysaraswat/brain-tumor-detector.git
   cd brain-tumor-detector
   pip install -r requirements.txt
   jupyter notebook TumorClassification.ipynb
   from tensorflow.keras.models import load_model
   model = load_model('model/tumor_model.keras')

---

## Requirements
Python 3.x
TensorFlow
Keras
NumPy
Matplotlib

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contributing
Feel free to submit pull requests to enhance the project, fix bugs, or add new features.

---



