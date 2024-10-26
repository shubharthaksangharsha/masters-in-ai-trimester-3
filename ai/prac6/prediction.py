import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


test_dir = os.path.join(base_dir, 'test')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    shuffle=False)  # Disable shuffling to align predictions with actual labels


# Load the saved model
loaded_model = tf.keras.models.load_model('cats_dogs_model_with_dropout.h5')

# Predict using the trained model
predictions = loaded_model.predict(test_generator, steps=len(test_generator), verbose=1)
#predictions = model_drop_out.predict(test_generator, steps=len(test_generator), verbose=1)

# Since the predictions are probabilities (between 0 and 1), we round them to get binary classification (0 or 1)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]


true_classes = test_generator.classes  # Actual labels from the test data
class_labels = list(test_generator.class_indices.keys())  # Get the class labels (e.g., ['cats', 'dogs'])


import matplotlib.pyplot as plt
import numpy as np

# Get the images and their corresponding labels
test_images, test_labels = next(test_generator)

# Plot a few test images along with the predicted and true labels
plt.figure(figsize=(10, 10))
for i in range(9):  # Plot first 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    plt.axis('off')
    
    # Set the title: Predicted and Actual label
    predicted_label = 'Dog' if predicted_classes[i] == 1 else 'Cat'
    true_label = 'Dog' if true_classes[i] == 1 else 'Cat'
    plt.title(f'Pred: {predicted_label}\nTrue: {true_label}', color=('green' if predicted_label == true_label else 'red'))

plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


