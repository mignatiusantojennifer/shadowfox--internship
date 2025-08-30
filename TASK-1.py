TASK -1 :
IMAGE TAGGING(CLAASSIFICATION) [DOG, CAT , CAR AND ETC]


import tensorflow as tf
from tensorflow.keras import layers, models

print("✅ TensorFlow version:", tf.__version__)

# Step 1: Load CIFAR-10 dataset (10 classes: airplane, car, bird, cat, etc.)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 2: Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")   # 10 output classes
])

# Step 3: Compile model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Step 4: Train model
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

# Step 5: Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc:.2f}")

# Step 6: Save model
model.save("cifar10_cnn_model.h5")
print("📁 Model saved as cifar10_cnn_model.h5")





OUTPUT : 

✅ TensorFlow version: 2.19.0
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step
/usr/local/lib/python3.12/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/3
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 66s 41ms/step - accuracy: 0.3999 - loss: 1.6566 - val_accuracy: 0.5935 - val_loss: 1.1576
Epoch 2/3
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 62s 39ms/step - accuracy: 0.6139 - loss: 1.1029 - val_accuracy: 0.6437 - val_loss: 1.0169
Epoch 3/3
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 63s 40ms/step - accuracy: 0.6741 - loss: 0.9296 - val_accuracy: 0.6747 - val_loss: 0.9359
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

🎯 Test Accuracy: 0.67
📁 Model saved as cifar10_cnn_model.h5
