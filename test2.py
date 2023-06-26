import os
import cv2
import dlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 替换为你的shape_predictor路径

# 遍历每个年龄组
age_groups = ["child", "teen", "adult", "senior"]

# 存储训练集和标签
train_features = []
train_labels = []

# 存储测试集和标签
test_features = []
test_labels = []

for i, age_group in enumerate(age_groups):
    images_folder = f"traindata/{age_group}"
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for rect in rects:
                shape = predictor(gray, rect)
                landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                if np.random.rand() < 0.8:  # 将80%的数据用作训练集
                    train_features.append(landmarks.flatten())
                    train_labels.append(i)
                else:  # 将20%的数据用作测试集
                    test_features.append(landmarks.flatten())
                    test_labels.append(i)

# 转换为NumPy数组
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# PCA降维
pca = PCA(n_components=2)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)

# SVM分类器
svm = SVC(kernel='linear')
svm.fit(train_features_pca, train_labels)

# 预测训练集和测试集
train_predictions = svm.predict(train_features_pca)
test_predictions = svm.predict(test_features_pca)

# 计算准确率
train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print("训练集准确率:", train_accuracy)
print("测试集准确率:", test_accuracy)

# 绘制混淆矩阵
confusion_mat = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=age_groups, yticklabels=age_groups)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 绘制SVM每一次迭代的准确率
plt.plot(svm.decision_function(train_features_pca), 'bo-', label='Train')
plt.plot(svm.decision_function(test_features_pca), 'ro-', label='Test')
plt.xlabel('Sample Index')
plt.ylabel('Decision Function Value')
plt.legend()
plt.title('SVM Decision Function')
plt.show()
