import tensorflow as tf
from tensorflow import keras

# Load the Fashion MNIST dataset
data = keras.datasets.fashion_mnist
(train_images, train_labels), (_, _) = data.load_data()

# Normalize the pixel values between 0 and 1
train_images = train_images / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Save the trained model
model.save('fashion_mnist_model.h5')
