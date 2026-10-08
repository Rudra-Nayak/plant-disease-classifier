"""Generate a dummy plant_model.pt for testing (untrained ResNet18 with 38 classes)."""
import json
import torch
import torch.nn as nn
from torchvision import models

with open("classes.json") as f:
    classes = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
torch.save(model.state_dict(), "plant_model.pt")
print(f"Saved dummy plant_model.pt with {len(classes)} classes")

