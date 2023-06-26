import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 设置文件夹路径和参数
train_dir = 'train_age_jpg'
test_dir = 'test_age_jpg'
batch_size = 16
epochs = 10

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
model.eval()  # 固定特征提取部分的参数，不进行训练

# 提取训练集图像特征
train_features = []
train_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        features = model.features(images)
        features = torch.flatten(features, start_dim=1)
        train_features.append(features)
        train_labels.extend(labels)

train_features = torch.cat(train_features, dim=0)
train_labels = torch.tensor(train_labels)

# 提取测试集图像特征
test_features = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        features = model.features(images)
        features = torch.flatten(features, start_dim=1)
        test_features.append(features)
        test_labels.extend(labels)

test_features = torch.cat(test_features, dim=0)
test_labels = torch.tensor(test_labels)

# 使用 SVM 进行分类
svm = SVC()
svm.fit(train_features.numpy(), train_labels.numpy())

# 在测试集上进行预测
predictions = svm.predict(test_features.numpy())
accuracy = accuracy_score(test_labels.numpy(), predictions)
print("Accuracy:", accuracy)
