# PRODIGY_ML_TASK-03
### Code Summary: Cat vs. Dog Image Classifier using SVM

1. **Setup and Configuration**:
   - Paths to training datasets for cats and dogs are defined (`cat_dir` and `dog_dir`).
   - Images are resized to 64x64 pixels (`IMG_SIZE`) for simplicity.

2. **Dataset Loading**:
   - **`load_data(cat_dir, dog_dir)`**:
     - Loads images from the specified directories.
     - Filters for valid image files (`.png`, `.jpg`, `.jpeg`).
     - Resizes images to 64x64 and normalizes pixel values to `[0, 1]`.
     - Assigns labels: `0` for cats and `1` for dogs.
     - Counts and reports the number of valid images loaded for each category.
   - Returns:
     - `data`: Numpy array of flattened image data.
     - `labels`: Corresponding labels.
     - Image counts for cats and dogs.

3. **Data Preprocessing**:
   - Normalizes pixel values (`data = data / 255.0`).
   - Flattens image data for compatibility with the SVM (`data.reshape(data.shape[0], -1)`).

4. **Training and Testing**:
   - Splits the dataset into training and testing sets (80% train, 20% test).
   - Trains a Support Vector Machine (SVM) classifier (`SVC`) with a linear kernel.
   - Evaluates the model using:
     - **Accuracy**: Percentage of correct predictions.
     - **Classification Report**: Precision, recall, and F1-score for each class.

5. **Image Classification**:
   - **`classify_image(image_path, model, img_size=64)`**:
     - Loads and preprocesses a single image (resize, normalize, and flatten).
     - Uses the trained SVM model to predict the class (cat or dog).
     - Returns "Cat" for label `0` and "Dog" for label `1`.

6. **Testing the Classifier**:
   - Classifies a single test image and prints the result.

### Output:
- Number of images loaded for cats and dogs.
- Model evaluation metrics: Accuracy and classification report.
- Classification result for a single test image.
