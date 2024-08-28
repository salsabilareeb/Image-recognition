# -*- coding: utf-8 -*-
"""
Areeb's MNIST Image Recognition Project

This script trains and evaluates different models on the MNIST dataset.
Models implemented:
1. Simple Neural Network (SNN)
2. Convolutional Neural Network (CNN)
3. Random Forest Classifier (RFC)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Load and preprocess the MNIST dataset
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

# Display sample images from the dataset
def display_sample_images(x_train, y_train):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap="gray")
        plt.xlabel(y_train[i])
    plt.show()

# Train a simple neural network model
def train_simple_nn(x_train, y_train, x_test, y_test):
    x_train_flattened = x_train.reshape(len(x_train), 28*28)
    x_test_flattened = x_test.reshape(len(x_test), 28*28)

    model = Sequential([Dense(10, input_shape=(784,), activation='sigmoid')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_flattened, y_train, epochs=10)
    
    test_loss, test_acc = model.evaluate(x_test_flattened, y_test)
    print('Simple NN Test accuracy:', test_acc)
    
    return model

# Train a Random Forest Classifier
def train_random_forest(x_train, y_train, x_test, y_test):
    x_train_flattened = x_train.reshape(len(x_train), 28*28)
    x_test_flattened = x_test.reshape(len(x_test), 28*28)

    rfc = RandomForestClassifier()
    rfc.fit(x_train_flattened, y_train)
    
    rfc_score = rfc.score(x_test_flattened, y_test)
    print("Random Forest accuracy:", rfc_score)
    
    return rfc

# Train a CNN model
def train_cnn(x_train, y_train, x_val, y_val):
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    return model

# Test the model on custom images
def test_on_custom_images(model, image_paths):
    for image_path in image_paths:
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
        image = image.resize((28, 28))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = 255 - image_array
        image_array = image_array / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
        
        prediction = model.predict(image_array)
        predicted_label = np.argmax(prediction)
        print(f"Predicted label for {image_path}: {predicted_label}")
        
        plt.imshow(image_array.reshape(28, 28), cmap="gray")
        plt.title(f"Predicted Label: {predicted_label}")
        plt.show()

# Main function to run the project
def main():
    # Load and preprocess the dataset
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

    # Display sample images from the training set
    display_sample_images(x_train, y_train)
    
    # Train and evaluate models
    simple_nn_model = train_simple_nn(x_train, y_train, x_test, y_test)
    cnn_model = train_cnn(x_train[:48000], y_train[:48000], x_train[48000:], y_train[48000:])
    rf_model = train_random_forest(x_train, y_train, x_test, y_test)

    # Test CNN on custom images
    custom_image_paths = ["/content/IMG_20240624_154610_161.jpg", "/content/IMG_20240624_154610_488.jpg"]
    test_on_custom_images(cnn_model, custom_image_paths)

if __name__ == "__main__":
    main()
