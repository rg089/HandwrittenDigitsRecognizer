import imutils
from cvTools.ConvNets.DigitNet import DigitNet
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data("mnist.npz")

X_train, X_test = imutils.normalize(X_train, X_test)
X_train, X_test = imutils.addDimension(X_train, X_test)
encoder, y_train, y_test = imutils.encodeY(y_train, y_test)

datagen = ImageDataGenerator(rotation_range=10,  zoom_range=0.20,  width_shift_range=0.1, height_shift_range=0.1)

checkpoint = ModelCheckpoint("models\\digitnet_mnist_augment_decay_5.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

epochs = 45

opt = Adam()
model = DigitNet.build(28, 28, 1, 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("Model Built.\nTraining model....")

history = model.fit(datagen.flow(X_train, y_train, batch_size=64), validation_data=(X_test, y_test), epochs=epochs,
                    batch_size=64, steps_per_epoch=X_train.shape[0]//64, verbose=1, callbacks=[checkpoint])

print("Evaluating Model...")
predictions = model.predict(X_test, batch_size=64)
print(classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), target_names=list(map(str, encoder.classes_))))

plt = imutils.plot_model(history, epochs)
plt.savefig("plots\\digitnet_training_mnist_augment_decay_5.png")
