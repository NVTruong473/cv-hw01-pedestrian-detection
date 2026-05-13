# Pedestrian Detection: From HOG + SVM to Deep Learning

Computer Vision Homework 01 — Pedestrian Detection using traditional Computer Vision and Deep Learning approaches.

This project explores the transition from **manual feature engineering** to **feature learning** for pedestrian detection and classification.

Methods implemented:

- HOG + SVM (Traditional Computer Vision)
- Simple ANN (Artificial Neural Network)
- MobileNetV2 Transfer Learning (Deep Learning)

---

## Project Overview

The goal of this homework is to compare different approaches for pedestrian detection and classification under the same Computer Vision pipeline.

The project investigates:

- Traditional handcrafted features (**HOG + SVM**)
- Basic neural network (**Simple ANN**)
- Transfer Learning using a pretrained CNN (**MobileNetV2**)

The comparison focuses on:

- Detection/classification performance
- Accuracy
- Inference speed (FPS)
- Robustness under difficult scenarios
- Failure cases

---

## Project Structure

```text
cv-hw01-pedestrian-detection/
│── notebook/
│   └── HW01_HOG_TransferLearning.ipynb
│
│── outputs/
│   ├── benchmark_images/
│   ├── prediction_results/
│   └── hog_benchmark.csv
│
│── models/
│   ├── ann_pedestrian.keras
│   └── mobilenetv2_pedestrian.keras
│
│── report/
│   └── report.pdf
│
│── README.md
```

---

## Dataset

### Deep Learning Dataset

Dataset source:

- **COCO 2017 Validation Split**
- Loaded using `tensorflow_datasets (tfds)`

Binary classification task:

```text
Person vs Non-Person
```

### Dataset Processing

Important preprocessing steps:

- COCO person class = `0`
- Dataset rebuilt using **large-person filtering**
- Image resize:

```text
64 × 128 × 3
```

- Pixel normalization:

```text
image / 255.0
```

Dataset split:

```text
Train      : 70%
Validation : 15%
Test       : 15%
```

---

## Part A — HOG + SVM

### Pipeline

```text
Input Image
    ↓
Resize
    ↓
HOG Feature Extraction
    ↓
SVM Classifier
    ↓
Sliding Window Detection
    ↓
Bounding Boxes
```

### Benchmark Parameters

The following parameters were tested:

### `winStride`

```python
(2,2)
(4,4)
(8,8)
```

### `scale`

```python
1.03
1.05
```

### Evaluation Metrics

- Number of pedestrians detected
- Inference time
- FPS (Frames Per Second)

### Traditional CV Strengths

- Fast inference
- Real-time potential
- No training required

### Limitations

- Sensitive to scale changes
- Sensitive to viewpoint changes
- Struggles under occlusion

---

## Part B — Deep Learning

### 1. Simple ANN

### Architecture

```text
Input Image
    ↓
Flatten
    ↓
Dense + ReLU
    ↓
Dense + ReLU
    ↓
Sigmoid
    ↓
Person / Non-Person
```

### Input / Output

Input:

```text
64 × 128 × 3 image
```

Output:

```text
Probability of person
```

### Observation

The ANN model showed weak performance because flattening removes spatial information.

This caused difficulty in learning human shapes and visual structures.

---

### 2. MobileNetV2 Transfer Learning

### Architecture

```text
Input Image
    ↓
MobileNetV2 (ImageNet pretrained)
    ↓
GlobalAveragePooling
    ↓
Dense Layer
    ↓
Dropout
    ↓
Sigmoid
    ↓
Person / Non-Person
```

### Why MobileNetV2?

MobileNetV2 was selected because:

- Lightweight architecture
- Fast inference
- Strong feature extraction
- Pretrained on ImageNet

### Transfer Learning Strategy

The pretrained backbone was frozen:

```python
base_model.trainable = False
```

Only the classifier head was retrained.

---

## Homework Image Testing

Test images:

```text
IMG_1.jpg
IMG_2.jpg
IMG_3.jpg
IMG_4.jpg
```

Evaluation was performed on unseen pedestrian images.

Observed challenges:

- Small pedestrians
- Crowded scenes
- Top-down viewpoints
- Occlusion

---

## Comparison

| Method | Type | Strength | Weakness |
|----------|------|-----------|-----------|
| HOG + SVM | Traditional CV | Fast detection | Sensitive to viewpoint |
| Simple ANN | Basic Deep Learning | Simple baseline | Lost spatial information |
| MobileNetV2 | Transfer Learning | Best performance | Can overfit |

---

## Results Summary

### HOG + SVM

- Benchmark on different `winStride` and `scale`
- FPS analysis
- Pedestrian detection comparison

### Simple ANN

- Weak performance
- Near-random classification behavior

### MobileNetV2

- Stronger generalization
- Better classification performance

---

## How to Run

### Install dependencies

```bash
pip install tensorflow
pip install tensorflow-datasets
pip install opencv-python
pip install imutils
pip install matplotlib
pip install scikit-learn
```

### Run notebook

Open:

```text
HW01_HOG_TransferLearning.ipynb
```

Run all cells sequentially.

---

## Report

The complete report is available in:

```text
report/report.pdf
```

---

## Limitations

Current limitations include:

- Small pedestrian detection
- Crowded scene recognition
- Domain gap between dataset and test images
- Limited training dataset size

---

## Future Improvements

Possible future work:

- Fine-tuning MobileNetV2
- YOLO-based pedestrian detection
- Better pedestrian cropping
- Larger training dataset
- Data augmentation

---

## References

1. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection.

2. CS231n: Convolutional Neural Networks for Visual Recognition

3. TensorFlow Keras Documentation

4. OpenCV HOGDescriptor Documentation

5. MobileNetV2 Paper
