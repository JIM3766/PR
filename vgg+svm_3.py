import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置文件夹路径和参数
train_dir = 'all_jpg/train'
test_dir = 'test_age_jpg'
batch_size = 16
epochs = 15

# 定义图像预处理和数据加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 VGG16 模型
model = vgg16(pretrained=True)

# 替换最后的全连接层
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, len(train_dataset.classes))

# 将模型参数设置为可微调
for param in model.features.parameters():
    param.requires_grad = True

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_accuracy_list = []
test_accuracy_list = []
train_loss_list = []
test_loss_list = []
# 迭代训练过程
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100.0 * correct / total
    train_accuracy_list.append(epoch_accuracy)
    train_loss_list.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # 在每个 epoch 结束后对测试集进行评估
    model.eval()
    test_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            test_labels.extend(labels.tolist())
    epoch_test_accuracy = accuracy_score(test_labels, predictions)
    test_accuracy_list.append(epoch_test_accuracy)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Epoch [{epoch+1}/{epochs}], Test Accuracy: {accuracy:.4f}")

