TASK-2 :
BOSTON HOUSE PRICE PREDICTION 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize

# 2. Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# 6. Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





OUTPUT : 
Epoch 1/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 66s 41ms/step - accuracy: 0.3595 - loss: 1.7377 - val_accuracy: 0.5680 - val_loss: 1.2081
Epoch 2/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 62s 40ms/step - accuracy: 0.5784 - loss: 1.1793 - val_accuracy: 0.6181 - val_loss: 1.0623
Epoch 3/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 84s 41ms/step - accuracy: 0.6385 - loss: 1.0201 - val_accuracy: 0.6291 - val_loss: 1.0611
Epoch 4/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 80s 40ms/step - accuracy: 0.6785 - loss: 0.9121 - val_accuracy: 0.6729 - val_loss: 0.9501
Epoch 5/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 87s 43ms/step - accuracy: 0.7056 - loss: 0.8359 - val_accuracy: 0.6841 - val_loss: 0.9169
Epoch 6/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 62s 40ms/step - accuracy: 0.7301 - loss: 0.7788 - val_accuracy: 0.6996 - val_loss: 0.8686
Epoch 7/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 62s 40ms/step - accuracy: 0.7453 - loss: 0.7279 - val_accuracy: 0.7113 - val_loss: 0.8562
Epoch 8/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 81s 40ms/step - accuracy: 0.7605 - loss: 0.6753 - val_accuracy: 0.7037 - val_loss: 0.8794
Epoch 9/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 83s 40ms/step - accuracy: 0.7799 - loss: 0.6249 - val_accuracy: 0.7082 - val_loss: 0.8606
Epoch 10/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 82s 40ms/step - accuracy: 0.7905 - loss: 0.5895 - val_accuracy: 0.7149 - val_loss: 0.8813
313/313 - 4s - 14ms/step - accuracy: 0.7149 - loss: 0.8813

✅ Test accuracy: 0.7149
