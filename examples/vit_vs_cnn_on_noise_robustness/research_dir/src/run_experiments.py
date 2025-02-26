import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets as torch_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import timm
from itertools import product

# Load CIFAR-10 dataset
test_dataset = datasets.load_dataset("cifar10")["test"]

# Define noise functions
def add_gaussian_noise(image, intensity):
    img = np.array(image)
    noise = np.random.normal(0, intensity, img.shape)
    noisy_img = img + noise
    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

def add_speckle_noise(image, intensity):
    img = np.array(image)
    noise = np.random.gamma(1, intensity, img.shape)
    noisy_img = img * noise
    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

def add_label_noise(label, intensity):
    if np.random.rand() < intensity:
        return np.random.randint(0, 10)
    return label

# Create noisy versions of the test set
noise_types = ["gaussian", "speckle", "label"]
intensities = [0.0, 0.1, 0.2, 0.5, 0.8]

# Define data transforms
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load clean CIFAR-10 train dataset
train_dataset = torch_datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define models
def get_vgg19():
    model = models.vgg19(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, 10)
    return model

def get_vit():
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
    return model

# Training function
def train_model(model, device, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Evaluation function
def evaluate_model(model, device, test_set, noise_type=None, intensity=0):
    model.eval()
    total_preds = []
    total_labels = []
    
    if noise_type:
        noisy_test = test_set.map(lambda x: {
            "img": add_gaussian_noise(x["img"], intensity) if noise_type == "gaussian" else
            add_speckle_noise(x["img"], intensity) if noise_type == "speckle" else
            x["img"],
            "label": add_label_noise(x["label"], intensity) if noise_type == "label" else x["label"]
        }, batched=False)
    else:
        noisy_test = test_set

    def collate_fn(batch):
        images = []
        labels = []
        for img, label in batch:
            tensor = data_transforms(img)
            images.append(tensor)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

    test_loader = torch.utils.data.DataLoader(
        noisy_test,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')
    
    return accuracy, precision, recall, f1

# Main experiment loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment 1: Basic Sensitivity Analysis
print("Starting Experiment 1: Basic Sensitivity Analysis")
vgg19 = get_vgg19()
vit = get_vit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(vgg19.parameters()) + list(vit.parameters()), lr=0.001)

# Train models
train_model(vgg19, device, train_loader, optimizer, criterion)
train_model(vit, device, train_loader, optimizer, criterion)

results = {}
for noise_type in noise_types:
    for intensity in intensities:
        print(f"Evaluating {noise_type} noise with intensity {intensity}")
        vgg_acc, vgg_pre, vgg_rec, vgg_f1 = evaluate_model(vgg19, device, test_dataset, noise_type, intensity)
        vit_acc, vit_pre, vit_rec, vit_f1 = evaluate_model(vit, device, test_dataset, noise_type, intensity)
        
        results[f"{noise_type}_{intensity}"] = {
            "VGG19": (vgg_acc, vgg_pre, vgg_rec, vgg_f1),
            "ViT": (vit_acc, vit_pre, vit_rec, vit_f1)
        }

# Generate figures
plt.figure(figsize=(10, 6))
for model in ["VGG19", "ViT"]:
    accuracies = [results[f"{noise}_{intensity}"][model][0] for noise, intensity in product(noise_types, intensities)]
    plt.plot(intensities, accuracies, label=model)
plt.xlabel("Intensity")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Noise Intensity")
plt.legend()
plt.savefig("Figure_1.png")
plt.close()

# Experiment 2: Impact of Pre-training and Data Augmentation
print("Starting Experiment 2: Impact of Pre-training and Data Augmentation")
# Use pre-trained models directly
vgg19_pretrained = get_vgg19()
vit_pretrained = get_vit()

# Evaluate with data augmentation
results_pretrained = {}
for noise_type in noise_types:
    for intensity in intensities:
        print(f"Evaluating {noise_type} noise with intensity {intensity} (Pre-trained)")
        vgg_acc, vgg_pre, vgg_rec, vgg_f1 = evaluate_model(vgg19_pretrained, device, test_dataset, noise_type, intensity)
        vit_acc, vit_pre, vit_rec, vit_f1 = evaluate_model(vit_pretrained, device, test_dataset, noise_type, intensity)
        
        results_pretrained[f"{noise_type}_{intensity}"] = {
            "VGG19": (vgg_acc, vgg_pre, vgg_rec, vgg_f1),
            "ViT": (vit_acc, vit_pre, vit_rec, vit_f1)
        }

# Generate figures
plt.figure(figsize=(10, 6))
for model in ["VGG19", "ViT"]:
    accuracies = [results_pretrained[f"{noise}_{intensity}"][model][0] for noise, intensity in product(noise_types, intensities)]
    plt.plot(intensities, accuracies, label=model)
plt.xlabel("Intensity")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Noise Intensity (Pre-trained)")
plt.legend()
plt.savefig("Figure_2.png")
plt.close()