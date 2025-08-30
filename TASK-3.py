TASK-3 :
AUTOCORRECT KEYBOARD SYSTEM :


# Install required libraries 
# !pip install tensorflow

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Load dataset (MNIST handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0–255 -> 0–1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels (0–9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Flatten 28x28 images into 784 vector
    Dense(128, activation='relu'),       # Hidden layer with ReLU
    Dense(64, activation='relu'),        # Another hidden layer
    Dense(10, activation='softmax')      # Output layer (10 classes)
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
print("\nTraining the Neural Network...")
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=2)

# 5. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# 6. Make predictions (example: first 5 test images)
predictions = model.predict(x_test[:5])
print("\nPredictions for first 5 test samples:")
print(predictions.argmax(axis=1))  # predicted labels
