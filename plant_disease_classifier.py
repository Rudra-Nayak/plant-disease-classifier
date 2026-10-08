"""
Plant Disease Classifier - Training Script
Trains a ResNet18 model on the PlantVillage dataset and saves:
  - plant_model.pt   (model state dict)
  - classes.json      (ordered list of class names)

Usage:
    python plant_disease_classifier.py --data_dir <path_to_plantvillage>
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_transforms():
    """Return train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model(num_classes: int) -> nn.Module:
    """Build a ResNet18 model with a custom classifier head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(data_dir: str, epochs: int = 10, batch_size: int = 32,
          lr: float = 1e-3, output_dir: str = ".", num_workers: int = 2):
    """Train the model and save weights + class list."""
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    model = build_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation phase ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "plant_model.pt"))
            print(f"  -> Saved best model (val_acc={best_acc:.4f})")

    # Save class names
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Model saved to {os.path.join(output_dir, 'plant_model.pt')}")
    print(f"Classes saved to {os.path.join(output_dir, 'classes.json')}")

    # Evaluate best model on validation set
    print("\nGenerating final evaluation report and confusion matrix...")
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "plant_model.pt"), map_location=device)
    )
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    from sklearn.metrics import classification_report
    from eval_report import save_confusion_matrix

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    save_confusion_matrix(
        all_labels, all_preds, class_names,
        path=os.path.join(output_dir, "confusion_matrix.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train plant disease classifier (ResNet18)")
    parser.add_argument("--data_dir", type=str, default="PlantVillage",
                        help="Path to dataset with train/ and val/ subdirs (default: PlantVillage)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr,
          args.output_dir, args.num_workers)
