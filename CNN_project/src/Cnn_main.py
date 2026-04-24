import os
import argparse
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms
from torchvision import models

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_dataloaders(batch_size=64, num_workers=2, fast_mode=False):
    """
    CIFAR-10:
    - 50,000 train images
    - 10,000 test images
    - RGB, 32x32
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    if fast_mode:
        train_dataset = Subset(train_dataset, range(10000))
        test_dataset = Subset(test_dataset, range(2000))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# Model 1: LeNet-like CNN

class LeNetLikeCNN(nn.Module):
    """
    LeNet-5 benzeri temel CNN.
    CIFAR-10 RGB olduğu için input channel = 3.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),      # 32x32 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),         # 28x28 -> 14x14

            nn.Conv2d(6, 16, kernel_size=5),     # 14x14 -> 10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)          # 10x10 -> 5x5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# Model 2: Improved CNN with BatchNorm + Dropout
# Same main hyperparameters as Model 1, but improvement layers added.

class ImprovedCNN(nn.Module):
    """
    İlk CNN'e benzer yapı korunur.
    İyileştirme olarak BatchNorm2d ve Dropout eklenmiştir.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Model 3 / Model 5: Ready CNN architecture from torchvision

def get_resnet18(num_classes=10, pretrained=False):
    """
    Literatürde yaygın kullanılan CNN mimarisi: ResNet18.
    CIFAR-10 için son fully connected katman 10 sınıfa göre değiştirilir.
    """
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Training / Evaluation

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def train_model(model, train_loader, test_loader, device, epochs, lr, model_name, out_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    best_acc = 0.0
    best_path = out_dir / f"{model_name}_best.pth"

    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, test_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        print(
            f"[{model_name}] Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    elapsed = time.time() - start

    # final evaluation
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    save_curves(history, model_name, out_dir)
    save_confusion_matrix(y_true, y_pred, model_name, out_dir)
    save_classification_report(y_true, y_pred, model_name, out_dir)

    return {
        "model": model_name,
        "best_test_acc": best_acc,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "train_time_sec": elapsed
    }




def save_curves(history, model_name, out_dir):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.savefig(out_dir / f"{model_name}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy Curve")
    plt.legend()
    plt.savefig(out_dir / f"{model_name}_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(CIFAR10_CLASSES)), CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(np.arange(len(CIFAR10_CLASSES)), CIFAR10_CLASSES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close()


def save_classification_report(y_true, y_pred, model_name, out_dir):
    report = classification_report(
        y_true,
        y_pred,
        target_names=CIFAR10_CLASSES,
        digits=4
    )

    with open(out_dir / f"{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)



# Hybrid Model: CNN Feature Extractor + Classical ML

def extract_features(model, loader, device):
    """
    ResNet18'in son sınıflandırıcı katmanı çıkarılır.
    CNN özellik çıkarıcı olarak kullanılır.
    Çıkan feature set ve label set .npy olarak kaydedilir.
    """
    model.eval()
    model = model.to(device)

    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features = feature_extractor(images)
            features = features.view(features.size(0), -1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


def run_hybrid_model(train_loader, test_loader, device, out_dir, fast_mode=False):
    print("\n[Hybrid] Extracting CNN features using ResNet18...")

    # CNN feature extractor
    cnn = get_resnet18(num_classes=10, pretrained=False)
    cnn = cnn.to(device)

    X_train, y_train = extract_features(cnn, train_loader, device)
    X_test, y_test = extract_features(cnn, test_loader, device)

    np.save(out_dir / "hybrid_X_train_features.npy", X_train)
    np.save(out_dir / "hybrid_y_train_labels.npy", y_train)
    np.save(out_dir / "hybrid_X_test_features.npy", X_test)
    np.save(out_dir / "hybrid_y_test_labels.npy", y_test)

    print(f"[Hybrid] X_train shape: {X_train.shape}")
    print(f"[Hybrid] y_train shape: {y_train.shape}")
    print(f"[Hybrid] X_test shape: {X_test.shape}")
    print(f"[Hybrid] y_test shape: {y_test.shape}")

    # For speed, RandomForest is generally faster than SVM for this emergency project.
    clf = RandomForestClassifier(
        n_estimators=100 if not fast_mode else 30,
        random_state=42,
        n_jobs=-1
    )

    print("[Hybrid] Training RandomForest on CNN features...")
    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    save_confusion_matrix(y_test, y_pred, "Hybrid_ResNet18_Features_RandomForest", out_dir)
    save_classification_report(y_test, y_pred, "Hybrid_ResNet18_Features_RandomForest", out_dir)

    with open(out_dir / "hybrid_feature_shapes.txt", "w", encoding="utf-8") as f:
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"y_train shape: {y_train.shape}\n")
        f.write(f"X_test shape: {X_test.shape}\n")
        f.write(f"y_test shape: {y_test.shape}\n")
        f.write(f"Hybrid RandomForest Accuracy: {acc:.4f}\n")

    return {
        "model": "Hybrid_ResNet18_Features_RandomForest",
        "best_test_acc": acc,
        "final_test_acc": acc,
        "final_test_loss": None,
        "train_time_sec": elapsed
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--skip_hybrid", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(42)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fast_mode=args.fast_mode
    )

    results = []

    # Model 1
    model1 = LeNetLikeCNN(num_classes=10)
    results.append(
        train_model(
            model1, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr,
            model_name="Model1_LeNetLikeCNN",
            out_dir=out_dir
        )
    )

    # Model 2
    model2 = ImprovedCNN(num_classes=10)
    results.append(
        train_model(
            model2, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr,
            model_name="Model2_ImprovedCNN_BN_Dropout",
            out_dir=out_dir
        )
    )

    # Model 3: Ready literature CNN architecture
    model3 = get_resnet18(num_classes=10, pretrained=False)
    results.append(
        train_model(
            model3, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr,
            model_name="Model3_ResNet18",
            out_dir=out_dir
        )
    )

    # Model 4: Hybrid CNN feature extractor + ML model
    if not args.skip_hybrid:
        results.append(
            run_hybrid_model(train_loader, test_loader, device, out_dir, fast_mode=args.fast_mode)
        )

    # Model 5: Full CNN model comparison
    # Since Model 3 already uses a full CNN architecture with the same CIFAR-10 dataset,
    # it can also be used as the full CNN comparison model against the hybrid method.
    # For explicitness, we save a note.
    with open(out_dir / "model5_note.txt", "w", encoding="utf-8") as f:
        f.write(
            "Model 5 requirement is satisfied by using Model3_ResNet18 as the full CNN architecture "
            "on the same CIFAR-10 dataset and comparing it with Model4 hybrid CNN-feature + RandomForest model.\n"
        )

    # Save summary
    summary_path = out_dir / "summary_results.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("========================\n\n")
        for r in results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Best Test Accuracy: {r['best_test_acc']:.4f}\n")
            f.write(f"Final Test Accuracy: {r['final_test_acc']:.4f}\n")
            f.write(f"Final Test Loss: {r['final_test_loss']}\n")
            f.write(f"Training Time Sec: {r['train_time_sec']:.2f}\n")
            f.write("------------------------\n")

    print("\nAll results saved into outputs/ folder.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
