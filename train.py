import torch
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from torchvision import datasets
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

# تعريف/استدعاء ال سي ان ان وتعديلها على حسب عدد الداتا
Num_class = 33
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(features, Num_class)

# adding Dropout
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(features, Num_class))

# نفعّل التدريب الكامل لجميع الطبقات (Fine-tuning)
for param in model.features.parameters():
    param.requires_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print("Using device:", device)

# ::::::::::::::::: hyperparameters :::::::::::::
learning_rate = 0.0001
num_epochs = 50
batch_size = 32
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

#::::::::::: preprocessing :::::::::::
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# تحميل البيانات من المجلدات
train_dataset = datasets.ImageFolder(root="ASLAD-Pad1.5Res1024MinCon0.5ModComp1FinallSplitted/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root="ASLAD-Pad1.5Res1024MinCon0.5ModComp1FinallSplitted/val", transform=val_test_transform)
test_dataset = datasets.ImageFolder(root="ASLAD-Pad1.5Res1024MinCon0.5ModComp1FinallSplitted/test",transform=val_test_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("train image number", len(train_dataset))
print("val image number", len(val_dataset))
print("test image number", len(test_dataset))
print("number of classes", len(train_dataset.classes))

# :::::::::::::تعريف متغيرات::::::::::::::
best_val_acc = 0
no_change_counter = 0
previous_acc = None
early_stopping = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

# validation Loop
    model.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_loss = val_running_loss / len(val_loader)
    scheduler.step(val_accuracy)

    # حفظ أفضل موديل
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), "Final_best_efficientnet_ASLAD.pth")
    print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Best: {best_val_acc:.2f}%")

    # توقف مبكر في حال استقرار المودل
    if previous_acc is not None:
        if previous_acc == val_accuracy:
            no_change_counter += 1
        else:
            no_change_counter = 0
    previous_acc = val_accuracy

    if no_change_counter >= early_stopping:
        print(f" no more improvment after [{epoch + 1}/{num_epochs}] Epoch")
        break

# test loop
model.load_state_dict(torch.load("Final_best_efficientnet_ASLAD.pth"))
model.eval()
correct = 0
total = 0
total_labels = []
total_preds = []
inference_times = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        start_time = time.time()
        outputs = model(images)
        end_time = time.time()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_labels.extend(labels.cpu().numpy())
        total_preds.extend(predicted.cpu().numpy())

        delay_for_each_btch = (end_time - start_time) / labels.size(0)
        inference_times.append(delay_for_each_btch)

cm = confusion_matrix(total_labels, total_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

test_accuracy = 100 * correct / total
precision = precision_score(total_labels, total_preds, average="macro") * 100
recall = recall_score(total_labels, total_preds, average="macro") * 100
f1 = f1_score(total_labels, total_preds, average="macro") * 100
avg_delay = sum(inference_times) / len(inference_times)
print(f"Test Accuracy : {test_accuracy:.2f}%")
print(f"Precision     : {precision:.2f}%")
print(f"Recall        : {recall:.2f}%")
print(f"F1 Score      : {f1:.2f}%")
print(f"Average Delay : {avg_delay * 1000:.2f} ms per image")
print("Model saved successfully!")
