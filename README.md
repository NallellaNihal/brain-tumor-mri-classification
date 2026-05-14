# Brain Tumor MRI Classification

A deep learning project for automatic brain tumor classification from MRI images using TensorFlow and Transfer Learning.

## Features

- Brain tumor MRI classification
- Multiple architectures supported
  - Enhanced Custom CNN
  - VGG19
  - InceptionV3
- Training, evaluation, and prediction modes
- High accuracy classification system
- TensorFlow/Keras implementation

---

## Dataset Classes

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 95.18% |
| AUC Score | 99.56% |

### Class-wise F1 Score

| Class | F1 Score |
|---|---|
| Glioma | 93.71% |
| Meningioma | 90.88% |
| No Tumor | 98.17% |
| Pituitary | 97.40% |

---

## Project Structure

```text
brain_tumor_mri_project/
│
├── configs/
├── data/
├── outputs/
├── src/
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Linux/WSL:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

```text
data/raw/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/
```

---

## Training

### Custom CNN

```bash
python main.py --mode train
```

### InceptionV3

```bash
python main.py --mode train --architecture inceptionv3_cnn
```

### VGG19

```bash
python main.py --mode train --architecture vgg19_cnn
```

---

## Evaluation

```bash
python main.py --mode evaluate --weights outputs/best_model.keras
```

---

## Prediction

```bash
python main.py --mode predict --weights outputs/best_model.keras --image data/raw/glioma/sample.jpg
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

---

## Future Improvements

- Web deployment using Flask/Streamlit
- Real-time MRI upload interface
- Better medical image preprocessing
- Multi-class segmentation
- GPU optimization

---

## Author

Developed as a Deep Learning Medical Imaging Project.