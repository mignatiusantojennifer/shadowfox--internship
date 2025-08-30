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



OUTPUT : 
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
/usr/local/lib/python3.12/dist-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)

Training the Neural Network...
Epoch 1/5
1688/1688 - 7s - 4ms/step - accuracy: 0.9258 - loss: 0.2564 - val_accuracy: 0.9693 - val_loss: 0.1090
Epoch 2/5
1688/1688 - 7s - 4ms/step - accuracy: 0.9678 - loss: 0.1059 - val_accuracy: 0.9727 - val_loss: 0.0935
Epoch 3/5
1688/1688 - 10s - 6ms/step - accuracy: 0.9769 - loss: 0.0741 - val_accuracy: 0.9763 - val_loss: 0.0841
Epoch 4/5
1688/1688 - 6s - 4ms/step - accuracy: 0.9829 - loss: 0.0547 - val_accuracy: 0.9778 - val_loss: 0.0758
Epoch 5/5
1688/1688 - 10s - 6ms/step - accuracy: 0.9856 - loss: 0.0445 - val_accuracy: 0.9805 - val_loss: 0.0746

✅ Test Accuracy: 97.76%
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step

Predictions for first 5 test samples:
[7 2 1 0 4]
