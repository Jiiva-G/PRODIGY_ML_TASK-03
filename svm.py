import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Set paths to dataset
cat_dir = "D:/ML_Intern_Task/Task-3/dataset/Train_dataset/cat"
dog_dir = "D:/ML_Intern_Task/Task-3/dataset/Train_dataset/dog"

# Image parameters
IMG_SIZE = 64  # Resize images to 64x64 for simplicity

def load_data(cat_dir, dog_dir):
    data = []
    labels = []
    cat_count = 0
    dog_count = 0

    # Load cat images
    for img_name in os.listdir(cat_dir):
        img_path = os.path.join(cat_dir, img_name)

        # Check if the file is an image
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)  # Label for cats
        cat_count += 1

    # Load dog images
    for img_name in os.listdir(dog_dir):
        img_path = os.path.join(dog_dir, img_name)

        # Check if the file is an image
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Check if the image was loaded successfully
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)  # Label for dogs
        dog_count += 1

    print(f"Number of cat images: {cat_count}")
    print(f"Number of dog images: {dog_count}")
    return np.array(data), np.array(labels), cat_count, dog_count

# Load the dataset
data, labels, cat_count, dog_count = load_data(cat_dir, dog_dir)

# Normalize pixel values and flatten the images
data = data / 255.0
data = data.reshape(data.shape[0], -1)  # Flatten for SVM

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function to classify a single image
def classify_image(image_path, model, img_size=64):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Resize and preprocess the image
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = img.flatten().reshape(1, -1)  # Flatten and reshape for SVM

    # Predict using the model
    prediction = model.predict(img)
    return "Cat" if prediction[0] == 0 else "Dog"

# Test the classifier with a single image
test_image_path = "D:/ML_Intern_Task/Task-3/dataset/Test_dataset/cat/Image_10.jpg"  # Replace with the path to a test image
result = classify_image(test_image_path, svm_model)
if result:
    print(f"The image is classified as: {result}")