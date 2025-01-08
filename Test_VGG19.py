import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
test_dir = '/home/utkarsh/Desktop/Image_Processing/Project/Split_data/val'
best_model_path = "/home/utkarsh/Desktop/Image_Processing/Project/VGG-19/best_vgg19_model.pth"

batch_size = 24
image_size = (224, 224)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model_vgg19 = models.vgg19(pretrained=True) 
num_features_vgg19 = model_vgg19.classifier[6].in_features
model_vgg19.classifier[6] = torch.nn.Linear(num_features_vgg19, len(test_dataset.classes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_vgg19 = model_vgg19.to(device)
model_vgg19.load_state_dict(torch.load(best_model_path, map_location=device))
model_vgg19.eval()

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

train_features_vgg19 = np.load("/home/utkarsh/Desktop/Image_Processing/Project/VGG-19/train_features_vgg19.npy")
train_labels_vgg19 = np.load("/home/utkarsh/Desktop/Image_Processing/Project/VGG-19/train_labels_vgg19.npy")
test_features_vgg19, test_labels_vgg19 = extract_features(model_vgg19, test_loader, device)

classifiers = {
    "SVM (Linear)": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

def plot_confusion_matrix(conf_matrix, class_labels, model_name):
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    plt.yticks(range(len(class_labels)), class_labels)
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, conf_matrix[i, j], va='center', ha='center')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

for name, clf in classifiers.items():
    clf.fit(train_features_vgg19, train_labels_vgg19)
    predictions = clf.predict(test_features_vgg19)
    accuracy = accuracy_score(test_labels_vgg19, predictions)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(f"\n{name} Classification Report:")
    print(classification_report(test_labels_vgg19, predictions))
    conf_matrix = confusion_matrix(test_labels_vgg19, predictions)
    plot_confusion_matrix(conf_matrix, test_dataset.classes, name)
