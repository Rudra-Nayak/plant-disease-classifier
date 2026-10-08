"""
Fast Transfer Learning for Plant Disease Classifier.
Trains on CPU in ~1-2 minutes by precomputing ResNet-18 features
on a balanced subset of the PlantVillage dataset.
"""

import json
import os
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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


def get_balanced_indices(dataset, max_per_class=40):
    class_to_indices = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        class_to_indices[target].append(idx)
    
    selected_indices = []
    for cls, indices in class_to_indices.items():
        random.seed(42)
        sampled = random.sample(indices, min(len(indices), max_per_class))
        selected_indices.extend(sampled)
    return selected_indices


def extract_features(feature_extractor, loader):
    features_list = []
    labels_list = []
    feature_extractor.eval()
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            out = feature_extractor(imgs)
            out = torch.flatten(out, 1)
            features_list.append(out)
            labels_list.append(lbls)
            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f"  Extracted batch {i + 1}/{len(loader)}")
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


def main():
    data_dir = "PlantVillage"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    print(f"Loading dataset from '{data_dir}'...")
    train_tfm, val_tfm = get_transforms()
    full_train = datasets.ImageFolder(train_dir, transform=train_tfm)
    full_val = datasets.ImageFolder(val_dir, transform=val_tfm)

    class_names = full_train.classes
    num_classes = len(class_names)
    print(f"Identified {num_classes} classes.")

    # Select balanced subsets
    train_indices = get_balanced_indices(full_train, max_per_class=50)
    val_indices = get_balanced_indices(full_val, max_per_class=15)
    print(f"Sampled {len(train_indices)} train images (~50/class) and {len(val_indices)} val images.")

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_val, val_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)

    print("\nLoading pre-trained ResNet-18 backbone...")
    weights = models.ResNet18_Weights.DEFAULT
    base_model = models.resnet18(weights=weights)
    
    # Feature extractor is everything except the final fc layer
    modules = list(base_model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)

    print("\nExtracting feature embeddings from train images (one-time pass)...")
    train_features, train_labels = extract_features(feature_extractor, train_loader)
    print("\nExtracting feature embeddings from validation images...")
    val_features, val_labels = extract_features(feature_extractor, val_loader)

    print(f"\nFeature extraction complete: Train shape {train_features.shape}, Val shape {val_features.shape}")

    # Train linear classifier on top of features
    classifier = nn.Linear(512, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.005)

    epochs = 25
    batch_size = 64
    num_train = train_features.size(0)

    print("\nTraining classification head...")
    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(num_train)
        running_loss = 0.0
        correct = 0

        for i in range(0, num_train, batch_size):
            idx = perm[i:i + batch_size]
            b_feat, b_lbl = train_features[idx], train_labels[idx]

            optimizer.zero_grad()
            outputs = classifier(b_feat)
            loss = criterion(outputs, b_lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * b_feat.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == b_lbl).sum().item()

        train_acc = correct / num_train

        # Validation
        classifier.eval()
        with torch.no_grad():
            v_outputs = classifier(val_features)
            _, v_preds = torch.max(v_outputs, 1)
            val_acc = (v_preds == val_labels).sum().item() / val_features.size(0)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    print("\nBuilding final full model and saving...")
    # Reconstruct complete ResNet18 model with trained head and pre-trained backbone
    final_model = models.resnet18(weights=weights)
    final_model.fc = classifier
    
    torch.save(final_model.state_dict(), "plant_model.pt")
    with open("classes.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print("Success! Model saved to 'plant_model.pt' and classes to 'classes.json'.")


if __name__ == "__main__":
    main()
