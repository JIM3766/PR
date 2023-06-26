import os
import cv2
import numpy as np
from sklearn.svm import SVC
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
train_folder = "traindata"
train_images, train_labels = load_images_from_folder(train_folder)

# Extract HOG features for training data
hog_features_train = []
for image in train_images:
    hog_features = extract_hog_features(image)
    hog_features_train.append(hog_features)
X_train = np.array(hog_features_train)
y_train = np.array(train_labels)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

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
y_pred = svm_classifier.predict(X_test)

# Compute confusion matrix
class_names = ['adult', 'child', 'senior', 'teen']
confusion_mat = confusion_matrix(y_test, y_pred, labels=class_names)

# Plot confusion matrix
plot_confusion_matrix(confusion_mat, classes=class_names)
plt.show()
# Compute accuracy
accuracy = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
print("Accuracy:", accuracy)
