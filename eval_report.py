"""Confusion matrix helper for the plant disease classifier.

Usage (inside plant_disease_classifier.py, after the classification report):
    from eval_report import save_confusion_matrix
    save_confusion_matrix(labels, preds, classes)

Needs: pip install matplotlib scikit-learn numpy
"""
import os
import json
import matplotlib

matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


def save_confusion_matrix(labels, preds, classes, path="confusion_matrix.png", top_n=10):
    n = len(classes)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))

    # Row-normalize so each row shows "of the real X images, what fraction went where"
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized confusion matrix (validation set)")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # The most frequent mistakes, ignoring correct predictions on the diagonal
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(
        ((off[i, j], i, j) for i in range(n) for j in range(n) if off[i, j] > 0),
        reverse=True,
    )[:top_n]
    print("\nMost common mistakes:")
    for count, i, j in pairs:
        print(f"  {count:4d} x  true '{classes[i]}' -> predicted '{classes[j]}'")


def evaluate_model(model_path="plant_model.pt", classes_path="classes.json",
                   val_dir="PlantVillage/val", output_img="confusion_matrix.png",
                   batch_size=64, max_samples=None):
    """Run full evaluation of a saved model on the validation dataset."""
    with open(classes_path, "r") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {model_path} on {val_dir} using {device}...")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=val_tfm)
    if max_samples and max_samples < len(val_dataset):
        indices = list(range(0, len(val_dataset), len(val_dataset) // max_samples))[:max_samples]
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    print(f"Running inference on {len(val_dataset)} validation samples...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            if (i + 1) % 20 == 0 or (i + 1) == len(val_loader):
                print(f"  Processed batch {i + 1}/{len(val_loader)}")

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nOverall Validation Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4, zero_division=0))

    save_confusion_matrix(all_labels, all_preds, classes, path=output_img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate plant disease classifier and generate confusion matrix")
    parser.add_argument("--model_path", type=str, default="plant_model.pt")
    parser.add_argument("--classes_path", type=str, default="classes.json")
    parser.add_argument("--val_dir", type=str, default="PlantVillage/val")
    parser.add_argument("--output_img", type=str, default="confusion_matrix.png")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=1500,
                        help="Max validation samples to evaluate for speed (0 for entire val set)")
    args = parser.parse_args()

    samples = None if args.max_samples <= 0 else args.max_samples
    evaluate_model(
        model_path=args.model_path,
        classes_path=args.classes_path,
        val_dir=args.val_dir,
        output_img=args.output_img,
        batch_size=args.batch_size,
        max_samples=samples,
    )

