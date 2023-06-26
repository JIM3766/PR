import os
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    return hog_features.flatten()

def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label = subfolder
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path, 0)  # Load image as grayscale
                if img is not None:
                    images.append(img)
                    labels.append(label)
    return images, labels

# Load training data
train_folder = "all_jpg/train"
train_images, train_labels = load_images_from_folder(train_folder)

# Extract HOG features for training data
hog_features_train = []
for image in train_images:
    hog_features = extract_hog_features(image)
    hog_features_train.append(hog_features)
X_train = np.array(hog_features_train)
y_train = np.array(train_labels)

# Train Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train, y_train)

# Load test data
test_folder = "test_age_jpg"
test_images, test_labels = load_images_from_folder(test_folder)

# Extract HOG features for test data
hog_features_test = []
for image in test_images:
    hog_features = extract_hog_features(image)
    hog_features_test.append(hog_features)
X_test = np.array(hog_features_test)
y_test = np.array(test_labels)

# Predict labels for test data
y_pred = decision_tree_classifier.predict(X_test)

# Compute confusion matrix
class_names = ['adult', 'child', 'senior', 'teen']
confusion_mat = confusion_matrix(y_test, y_pred, labels=class_names)

# Plot confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = confusion_mat.max() / 2.
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        plt.text(j, i, format(confusion_mat[i, j], 'd'), ha="center", va="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
accuracy = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
print("Accuracy:", accuracy)