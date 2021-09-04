import pandas as pd
import numpy as np
from mnist import MNIST
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# read in training data
mndata = MNIST('data')
images_train, labels_train = mndata.load_training()
# read in testing data
images_test, labels_test = mndata.load_testing()

# print number from training data
# index = random.randrange(0, len(images_train))
# print(labels_train[index])
# print(mndata.display(images_train[index]))

# convert to numpy arrays and reshape
images_train_arr = np.array(images_train)
images_train_arr = images_train_arr.reshape(images_train_arr.shape[0], 28, 28, 1)
labels_train = np.array(labels_train)

images_test_arr = np.array(images_test)
images_test_arr = images_test_arr.reshape(images_test_arr.shape[0], 28, 28, 1)
labels_test = np.array(labels_test)

# normalize RBG values
images_train_arr = images_train_arr.astype('float32')
images_test_arr = images_test_arr.astype('float32')

images_train_arr /= 255
images_test_arr /= 255

# build a CNN
model = Sequential()
model.add(Conv2D(28, kernel_size = (3, 3), input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation = tf.nn.softmax))

# train model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x = images_train_arr, y = labels_train, epochs=10)

# test model
model.evaluate(images_test_arr, labels_test)

# make an individual prediction
index = random.randrange(0, len(images_test))
print(mndata.display(images_test[index]))
pred = model.predict(images_test_arr[index].reshape(1, 28, 28, 1))
print(pred.argmax())