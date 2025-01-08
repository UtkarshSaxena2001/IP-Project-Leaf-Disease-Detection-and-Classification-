import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared GPU cache.")
    
train_dir = '/home/utkarsh/Desktop/Image_Processing/Project/Split_data/train'
best_model_path = "/home/utkarsh/Desktop/Image_Processing/Project/ResNet50/best_resnet50_model.pth"
curve_path = "/home/utkarsh/Desktop/Image_Processing/Project/ResNet50/accuracy_loss_curve_resnet50.png"

batch_size = 45
image_size = (224, 224)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model_resnet50 = models.resnet50(pretrained=True)
num_features_resnet50 = model_resnet50.fc.in_features
model_resnet50.fc = torch.nn.Linear(num_features_resnet50, len(train_dataset.classes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_resnet50 = model_resnet50.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer_resnet50 = torch.optim.Adam(model_resnet50.parameters(), lr=0.001)

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Extracting Features")
        for images, lbls in progress_bar:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

def train_model_with_save(model, train_loader, criterion, optimizer, device, epochs=20, save_path=best_model_path, curve_path=curve_path):
    model.train()
    best_loss = float('inf')
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            progress_bar.set_postfix(loss=(running_loss / (i + 1)))

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with loss: {best_loss:.4f}")
            train_features, train_labels = extract_features(model, train_loader, device)
            np.save("/home/utkarsh/Desktop/Image_Processing/Project/ResNet50/train_features_resnet50.npy", train_features)
            np.save("/home/utkarsh/Desktop/Image_Processing/Project/ResNet50/train_labels_resnet50.npy", train_labels)
            print("Features and labels for the best model saved.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b', label="Training Loss")
    plt.plot(range(1, epochs + 1), epoch_accuracies, marker='s', linestyle='--', color='g', label="Training Accuracy")
    plt.title("Training Accuracy vs Loss Curve for ResNet50")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    print(f"Training accuracy and loss curve saved to: {curve_path}")

train_model_with_save(model_resnet50, train_loader, criterion, optimizer_resnet50, device, epochs=20, save_path=best_model_path, curve_path=curve_path)
