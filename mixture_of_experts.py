import numpy as np
from random import randint
import tensorflow as tf
import os
import warnings
from data_loader import ImageDataset
from datetime import datetime
import matplotlib.pyplot as plt
from cnn_classifier1 import CNN as BaseClassifier
from cnn_classifier2 import CNN as LeNet

class MoE():
    pass

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("TensorFlow Version: ", tf.__version__)

    path = "GTSRB/"
    gtsrb = ImageDataset(path)
    n_train = len(gtsrb.x_train)
    n_valid = len(gtsrb.x_valid)
    n_test = len(gtsrb.x_test)
    width, height = len(gtsrb.x_test[0]), len(gtsrb.x_test[0][0])
    image_shape = (width, height)
    n_classes = len(set(gtsrb.y_test))

    epochs = 50
    learning_rate = 1e-3

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    print()

    # gtsrb.display_one(gtsrb.x_train[0])

    start = datetime.now()
    base = BaseClassifier(gtsrb, num_epochs=epochs, learning_rate=learning_rate)
    lenet = LeNet(gtsrb, num_epochs=epochs, learning_rate=learning_rate)
    end = datetime.now()
    print("Time taken to build and train each model on " + str(epochs) + " epochs is:", str(end - start))
    base.tf_sess.close()
    lenet.tf_sess.close()
