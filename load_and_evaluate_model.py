import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the Fashion MNIST dataset
data = keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = data.load_data()
test_images = test_images / 255.0

# Load the saved model
loaded_model = keras.models.load_model('fashion_mnist_model.h5')

# Reshape test images to match the expected input shape of the model
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Use the loaded model for predictions
predictions = loaded_model.predict(test_images)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Compute accuracy score
accuracy = np.mean(predicted_labels == test_labels)
print('Accuracy:', accuracy)

# Plot confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
print('\nClassification Report:')
print(classification_report(test_labels, predicted_labels, target_names=class_names))
