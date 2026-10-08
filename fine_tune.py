"""
Fine-tuning script for Plant Disease Classifier.
Unfreezes ResNet-18 layer4 + fc with differential learning rates,
data augmentation, and balanced sampling across all 38 classes.
"""

import json
import os
import random
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def get_transforms():
    """Robust data augmentations for fine-tuning."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
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


def get_balanced_indices(dataset, max_per_class=100, seed=42):
    """Sample an evenly balanced subset across all classes."""
    class_to_indices = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        class_to_indices[target].append(idx)

    selected = []
    rng = random.Random(seed)
    for cls, indices in class_to_indices.items():
        k = min(len(indices), max_per_class)
        selected.extend(rng.sample(indices, k))
    return selected


def fine_tune(data_dir="PlantVillage", epochs=4, batch_size=48, samples_per_class=100):
    # Optimize PyTorch CPU threading
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(max(2, min(cpu_cores, 8)))
    print(f"Using {torch.get_num_threads()} CPU threads for fine-tuning.")

    train_tfm, val_tfm = get_transforms()
    full_train = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    full_val = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfm)

    class_names = full_train.classes
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes from '{data_dir}'.")

    # Sample balanced dataset
    train_indices = get_balanced_indices(full_train, max_per_class=samples_per_class, seed=42)
    val_indices = get_balanced_indices(full_val, max_per_class=25, seed=42)
    print(f"Sampled {len(train_indices)} training images and {len(val_indices)} validation images.")

    train_loader = DataLoader(
        Subset(full_train, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        Subset(full_val, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model: load existing trained weights if present
    print("Loading model weights...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if os.path.exists("plant_model.pt"):
        print("Starting from existing 'plant_model.pt' weights.")
        model.load_state_dict(torch.load("plant_model.pt", map_location="cpu", weights_only=True))
    else:
        print("Starting from ImageNet pre-trained weights.")
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.load_state_dict(base.state_dict(), strict=False)

    # Freeze conv1, bn1, layer1, layer2, layer3 (early feature extractors)
    # Unfreeze layer4 (high-level visual features: blight spots, leaf lesions, textures) and fc head
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Differential learning rates: lower for layer4, higher for classification head
    optimizer = optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": model.fc.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    total_train = len(train_indices)
    total_val = len(val_indices)

    print(f"\nStarting fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        t0 = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                pct = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Epoch {epoch+1:02d} [{batch_idx+1:02d}/{len(train_loader):02d} ({pct:.0f}%)] Loss: {loss.item():.4f}")

        train_loss = running_loss / total_train
        train_acc = correct / total_train

        # Validation phase
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbls).sum().item()

        val_acc = val_correct / total_val
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d}/{epochs:02d} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "plant_model.pt")
            print(f"  -> Saved new best model checkpoint (Val Acc: {best_val_acc*100:.2f}%)")

    # Update classes.json
    with open("classes.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nFine-tuning completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print("Model saved to 'plant_model.pt'.")


if __name__ == "__main__":
    fine_tune()
