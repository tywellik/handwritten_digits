import pandas as pd
import numpy as np
# import gzip
from mnist import MNIST
import random

# read in training data
mndata = MNIST('data')
images_train, labels_train = mndata.load_training()
# read in testing data
images_test, labels_test = mndata.load_testing()

# print number from training data
index = random.randrange(0, len(images_train))  # choose an index ;-)
print(labels_train[index])
print(mndata.display(images_train[index]))

# convert to numpy arrays and reshape
images_train = np.array(images_train)
images_train = images_train.reshape(images_train.shape[0], 28, 28, 1)
labels_train = np.array(labels_train)

images_test = np.array(images_test)
images_test = images_test.reshape(images_test.shape[0], 28, 28, 1)
labels_test = np.array(labels_test)

print(images_test.shape)
print(labels_test.shape)

