# FexG - Explainable Federated ECG Arrhythmia Classification

FexG is a privacy-preserving and explainable AI framework for ECG arrhythmia classification using Federated Learning and Deep Learning. The project combines distributed training with explainability techniques to classify ECG heartbeat signals while preserving patient data privacy.

---

## Features

- Federated Learning (FedAvg)
- Explainable AI (Integrated Gradients & Saliency Maps)
- ECG Arrhythmia Classification
- FastAPI Backend
- React Frontend
- MIT-BIH Arrhythmia Dataset
- Real-time Prediction & Visualization

---

## Tech Stack

### Backend
- Python
- FastAPI
- TensorFlow / Keras
- NumPy
- Scikit-learn

### Frontend
- React
- JavaScript
- Chart Visualization

---

## Dataset

- MIT-BIH Arrhythmia Database (PhysioNet)

---

## Supported Heartbeat Classes

- Normal Beat (N)
- Left Bundle Branch Block (L)
- Right Bundle Branch Block (R)
- Atrial Premature Beat (A)
- Ventricular Premature Beat (V)

---

## Model Performance

- Accuracy: **99.03%**
- High Precision, Recall, and F1-score across all classes

---

## Project Structure

```bash
website/
│
├── backend/
│   ├── xai.py
│   ├── models/
│   └── ...
│
├── frontend/
│   ├── src/
│   └── ...
│
├── mitbih_database/
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/avik-halder/FeXG.git
cd FeXG
```

### Backend Setup

```bash
cd backend
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### Run Backend

```bash
python -m fastapi dev xai.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

---

## Explainable AI

The framework uses:
- Integrated Gradients
- Saliency Maps

to highlight clinically important ECG waveform regions influencing model predictions.

---

## Research Focus

This project aims to:
- Preserve patient privacy
- Improve clinical trust in AI systems
- Enable interpretable federated healthcare AI

---

## Future Improvements

- Real-world federated deployment
- Differential Privacy
- Secure Aggregation
- Edge-device optimization
- Clinician-in-the-loop evaluation

---

## Authors

- Avik Halder
- Akibul Hasan Anik
- Hrithik Das
- Suman Saha

---

## License

This project is intended for research and educational purposes.