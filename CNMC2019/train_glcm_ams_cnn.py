import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import json

train_cnn_asm_flag = True
train_cnn_contrast_flag = False
train_cnn_dissimilarity_flag = False
train_cnn_energy_flag = False
train_cnn_entropy_flag = False
train_cnn_homogeneity_flag = False
train_cnn_max_flag = False
train_cnn_mean_flag = False
train_cnn_standard_flag = False


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_path =""
data_dir = ""
output_dir = ""
base_path = os.path.join("D:", "Research", "CNMC2019")
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directories
if train_cnn_asm_flag:
    data_dir = os.path.join(base_path, "GLCM_AMS")  # Root folder containing "all" and "hem" subfolders
    output_dir = os.path.join(current_dir, "CNN", "GLCM_AMS_Result")  # Directory to save results and models
elif train_cnn_contrast_flag:
    data_dir = os.path.join(base_path, "GLCM_Contrast")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Contrast_Result")
elif train_cnn_dissimilarity_flag:
    data_dir = os.path.join(base_path, "GLCM_Dissimilarity")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Dissimilarity_Result")
elif train_cnn_energy_flag:
    data_dir = os.path.join(base_path, "GLCM_Energy")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Energy_Result")
elif train_cnn_entropy_flag:
    data_dir = os.path.join(base_path, "GLCM_Entropy")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Entropy_Result")
elif train_cnn_homogeneity_flag:
    data_dir = os.path.join(base_path, "GLCM_Homogeneity")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Homogeneity_Result")
elif train_cnn_max_flag:
    data_dir = os.path.join(base_path, "GLCM_Max")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Max_Result")
elif train_cnn_mean_flag:
    data_dir = os.path.join(base_path, "GLCM_Mean")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Mean_Result")
elif train_cnn_standard_flag:
    data_dir = os.path.join(base_path, "GLCM_Standard")
    output_dir = os.path.join(current_dir, "CNN", "GLCM_Standard_Result")

os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 20
num_folds = 10

# Transforms
data_transforms = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Normalize to range [-1, 1]
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
num_classes = len(dataset.classes)  # Number of classes (binary: 2)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 87 * 87, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model Definition (easily replaceable)
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# K-Fold Cross-Validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Training Loop
fold_results = []
all_val_auc = []  # Initialize list to store roc_auc values

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Split dataset
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    # model = get_model(num_classes).to(device)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics storage
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "val_auc": []}

    best_val_f1 = 0.0
    best_model_path = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # For metrics
                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average="weighted")
        val_auc = roc_auc_score(all_targets, all_probs)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(output_dir, f"fold_{fold + 1}_best_model.pth")
            torch.save(model.state_dict(), best_model_path)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

    # Save fold results
    history_path = os.path.join(output_dir, f"fold_{fold + 1}_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved training history for fold {fold + 1} at {history_path}")

    # Save confusion matrix data
    cm_results = {"true_labels": all_targets, "predicted_labels": all_preds}
    cm_results_path = os.path.join(output_dir, f"fold_{fold + 1}_confusion_matrix_data.csv")
    pd.DataFrame(cm_results).to_csv(cm_results_path, index=False)
    print(f"Saved confusion matrix data for fold {fold + 1} at {cm_results_path}")

    # Record fold results summary
    fold_results.append({"fold": fold + 1, "val_f1": best_val_f1, "val_auc": val_auc})

    # Append roc_auc to list
    all_val_auc.append(history["val_auc"])  # Append roc_auc for current fold

# Save fold summary
fold_summary_path = os.path.join(output_dir, "fold_summary.csv")
pd.DataFrame(fold_results).to_csv(fold_summary_path, index=False)
print(f"Saved fold summary at {fold_summary_path}")

# Save all_val_auc list to file
with open(os.path.join(output_dir, "all_val_auc.json"), "w") as f:
    json.dump(all_val_auc, f)
print(f"Saved all_val_auc at {os.path.join(output_dir, 'all_val_auc.json')}")