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
