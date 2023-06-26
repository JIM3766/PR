import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的VGG16模型
vgg_model = models.vgg16(pretrained=True)
vgg_model.to(device)

# 设置图像的预处理转换
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练集数据
train_dataset = ImageFolder("traindata", transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 加载测试集数据
test_dataset = ImageFolder("test_age_jpg", transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 创建自定义的全连接层作为分类器
num_classes = len(train_dataset.classes)  # 类别数为训练集中的类别数

# 获取VGG16模型的特征提取部分（卷积层）
features = vgg_model.features

# 创建自定义的分类器
classifier = nn.Sequential(
    nn.Linear(25088, 4096),  # 添加第一层全连接层
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),  # 添加第二层全连接层
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes)  # 最后一层全连接层输出类别数
)

# 替换VGG16模型的分类器
vgg_model.classifier = classifier
vgg_model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器，将VGG16模型的所有参数都作为待优化参数
optimizer = torch.optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)

# 存储每个epoch的训练和测试结果
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 在每个epoch中进行训练和测试
epochs = 10

for epoch in range(epochs):
    # 训练模式
    vgg_model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = vgg_model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    # 计算训练集平均损失和准确率
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)

    # 测试模式
    vgg_model.eval()
    test_loss = 0.0
    test_correct = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = vgg_model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    # 计算测试集平均损失和准确率
    test_loss = test_loss / len(test_dataset)
    test_accuracy = test_correct / len(test_dataset)

    # 存储每个epoch的训练和测试结果
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 打印训练和测试结果
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

# 绘制每个epoch的训练和测试结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 计算最终测试集的混淆矩阵和ROC曲线
class_names = test_dataset.classes
confusion_mat = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Test Dataset')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

# ...

# 将预测的概率转换为预测的标签
_, predicted_labels = torch.max(outputs.data, 1)

# 将标签从张量转换为NumPy数组
true_labels = labels.cpu().numpy()
predicted_labels = predicted_labels.cpu().numpy()

# 计算ROC曲线和AUC值
fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_labels)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(['ROC curve (AUC = {:.2f})'.format(roc_auc)])
plt.show()

