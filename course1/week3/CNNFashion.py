import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = \
    mnist.load_data()

training_images = np.expand_dims(training_images, axis=3)
training_images = training_images / 255.0

test_images = np.expand_dims(test_images, axis=3)
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 128, activation = 'relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)