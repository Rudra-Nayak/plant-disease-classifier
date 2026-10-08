# 🌿 PlantGuard — Leaf Disease Classifier

A Streamlit web app that uses a ResNet-18 deep-learning model to identify
common plant leaf diseases from the
[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) dataset.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the model

The dataset is located in `PlantVillage/` containing 38 classes across `train/` and `val/`.

To train the ResNet-18 model on the full dataset:

```bash
python plant_disease_classifier.py --data_dir PlantVillage --epochs 10
```

This trains the network and produces `plant_model.pt` and `classes.json`.

For quick testing without running full training, generate a 38-class dummy model:

```bash
python create_dummy_model.py
```

### 3. Run the app

```bash
python -m streamlit run app.py
```
*(or `streamlit run app.py` if Streamlit is added to your PATH)*

The app opens at [http://localhost:8501](http://localhost:8501).

---

## Features

| Feature | Description |
|---|---|
| 📁 File Upload | Upload JPG / PNG / WebP leaf images |
| 📷 Camera Input | Snap a photo directly from your device camera |
| 🔍 Top-3 Predictions | Confidence bars for the three most likely diagnoses |
| ⚠️ Low-Confidence Alert | Warning when the model is less than 60% confident |
| 🎨 Glass UI | Dark gradient background with frosted-glass cards |
| 📱 Mobile Responsive | Layout adapts to narrow screen widths |

---

## Project Structure

```
plant classifier/
├── app.py                      # Streamlit frontend (main entry point)
├── plant_disease_classifier.py # Full training script (ResNet-18)
├── fast_train.py               # Fast CPU transfer learning script
├── eval_report.py              # Generates classification report & confusion matrix
├── confusion_matrix.png        # Evaluation confusion matrix plot
├── create_dummy_model.py       # Generates an untrained model for testing
├── plant_model.pt              # Trained model weights (generated)
├── classes.json                # Ordered class names (generated)
├── requirements.txt
└── README.md
```

---

## Dependencies

- **streamlit** — Web UI framework
- **pillow** — Image loading and processing
- **torch** — PyTorch deep-learning framework
- **torchvision** — Pre-trained models and image transforms
- **matplotlib**, **scikit-learn**, **numpy** — Evaluation metrics and confusion matrix plotting
