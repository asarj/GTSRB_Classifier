import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pickle

class ImageDataset():
    train = None
    x_train = None
    y_train = None

    test = None
    x_test = None
    y_test = None

    validation = None
    x_valid = None
    y_valid = None

    tf_sess = None
    shape = None
    num_classes = None
    batch_size = 256


    def __init__(self, dir):
        """Constructor for building the TensorFlow dataset"""
        print("TensorFlow Version: ", tf.__version__)
        self.tf_sess = tf.Session()
        self.load(dir)
        self.preprocess(self.train['features'])
        self.setup_batch_iterator(self.train['features'], self.train['labels'])

    def load(self, directory:str)->None:
        """Populates the train, test, and validation global variables with raw data from the pickled files"""

        self.train = pickle.load(open(directory + 'train.pickle', 'rb'))
        self.x_train, self.y_train = self.train['features'], self.train['labels']

        self.shape = self.x_train[0].shape
        print("Shape: ", self.shape)
        self.num_classes = len(np.unique(self.y_train))
        print("Unique Classes: ", self.num_classes)

        self.test = pickle.load(open(directory + 'test.pickle', 'rb'))
        self.x_test, self.y_test = self.test['features'], self.test['labels']

        self.validation = pickle.load(open(directory + 'valid.pickle', 'rb'))
        self.x_valid, self.y_valid = self.validation['features'], self.validation['labels']

    def display_one(self, a, title1 = "Original"):
        """Helper function for displaying an image"""

        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def display_two(self, a, b, title1="Original", title2="Edited"):
        """Helper function for displaying two images, usually for comparing before and after transformations"""

        plt.subplot(121)
        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(b)
        plt.title(title2)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def preprocess(self, features:np.ndarray)->None:
        """Main function for preprocessing images"""

        for img in features[:5]:
            print("Before ", img[0][0])
            # self.display_one(img)
            img = self.normalize_image_pixels(img)
            print("After ", img[0][0])
            # self.display_one(img)

    def normalize_image_pixels(self, image:np.ndarray)->np.ndarray:
        """Function to normalize the image pixels. Assumes that the np.ndarray passed in contains values
            from [0,255] and normalizes it down to a value that is [0, 1)"""

        return np.divide(image, 255.0)

    def setup_batch_iterator(self, features:np.ndarray, labels:np.ndarray):
        """Function to construct a TensorFlow dataset and set up the batch iterator"""
        pass

if __name__ == "__main__":
    path = "GTSRB/"
    gtsrb = ImageDataset(path)
    n_train = len(gtsrb.train['features'])
    n_valid = len(gtsrb.validation['features'])
    n_test = len(gtsrb.test['features'])
    width, height = len(gtsrb.test["features"][0]), len(gtsrb.test["features"][0][0])
    image_shape = (width, height)
    n_classes = len(set(gtsrb.test["labels"]))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    for image in gtsrb.train['features'][:5]:
        print("shape: {0}, min: {1}, max: {2}".format(
            image.shape, image.min(), image.max()))
    # print(type(gtsrb.train['features'][0]))