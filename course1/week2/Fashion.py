import tensorflow as tf
import matplotlib.pyplot as plt

mnist_fashion = tf.keras.datasets.fashion_mnist

(training_data, training_labels), (test_data, test_labels) = mnist_fashion.load_data()

training_data = training_data/255.0
test_data = test_data/255.0

# fig, ax = plt.subplots(3,3)
# for i in range(3):
#     for j in range(3):
#         ax[i,j].imshow(training_data[i*3+j])
#plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.8):
            print("Reach 80% accuracy so cancelling training")
            self.model.stop_training = True

#model.compile(optimizer = tf.keras.optimizers.Adam(),
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = myCallBack()
model.fit(training_data, training_labels, epochs=10, callbacks=[callbacks])
#model.evaluate(test_data, test_labels)