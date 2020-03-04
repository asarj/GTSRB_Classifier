import numpy as np
import tensorflow as tf
import os
import warnings
from data_loader import ImageDataset
from datetime import datetime

class CNN():

    tf_sess = None
    model = None
    dataset = None
    batch_size = 128
    repeat_size = 5
    shuffle = 128

    def __init__(self, dataset:ImageDataset, num_epochs=100):
        self.tf_sess = tf.Session()
        self.dataset = dataset
        # self.setup_batch_iterator()
        self.build_model()
        self.train_model(num_epochs)

    def build_model(self, learning_rate=0.001):
        print("Building model...")
        self.x = tf.placeholder(tf.float32, [None] + self.dataset.shape)
        self.y = tf.placeholder(tf.int32, [None])

        print("Shape of initial layer:", self.x.shape)

        # First layer
        c1_channels = 3
        c1_filters = 6
        c1 = self.conv_layer(input=self.x, input_channels=c1_channels, filters=c1_filters, filter_size=5)

        print("Shape of After 1st layer:", c1.shape)
        # Pooling
        pool1 = self.pool(layer=c1, ksize=[1,2,2,1], strides=[1,2,2,1])
        print("Shape of After 1st pooling:", pool1.shape)

        # Flattened layer
        flattened = self.flatten_layer(layer=pool1)
        print("Shape of After flattening:", flattened.shape)

        # First Fully Connected Layer
        fc1_input = 1176
        fc1_output = 500
        fc1 = self.fc_layer(input=flattened, inputs=fc1_input, outputs=fc1_output, relu=True)
        print("Shape of After 1st FC:", fc1.shape)

        # Logits
        l_inp = 500
        l_out = 43
        logits = self.fc_layer(input=fc1, inputs=l_inp, outputs=l_out, relu=False)
        print("Shape after logits:", logits.shape)

        # Convert train data labels to one hot encoding to feed into softmax function
        y_to_one_hot = tf.one_hot(self.y, self.dataset.num_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_to_one_hot)

        self.loss = tf.reduce_mean(cross_entropy)
        # print("Loss: ", self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # print("Optimizer:", self.optimizer)

        correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_to_one_hot, axis=1))
        # correct =
        # print(correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # print("Accuracy", self.accuracy)
        self.prediction = tf.argmax(logits, axis=1)
        # print("Prediction", self.prediction)

    def train_model(self, epochs:int):
        print("Training model...")
        self.tf_sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            self.tf_sess.run(self.dataset.train_init)
            try:
                total = 0
                while 1:
                    bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])
                    # print("Batch " + str(i + 1))
                    # print("X_batch:", bx.shape)
                    # print("Y_batch:", by.shape)
                    # print()
                    feed_dict = {
                        self.x: bx,#.reshape((-1, 32, 32, 3)),
                        self.y: by#.reshape((-1))
                    }
                    self.tf_sess.run(self.optimizer, feed_dict=feed_dict)
                    loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    total += acc * len(by)
                    # print(total / len(self.dataset.y_train))
            except(tf.errors.OutOfRangeError):
                pass


            feed_dict = {
                self.x: self.dataset.x_valid,
                self.y: self.dataset.y_valid
            }

            loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            print(f'epoch {epoch + 1}: loss = {loss:.4f}, training accuracy = {total / len(self.dataset.y_train):.4f}, validation accuracy = {acc:.4f}')

        feed_dict = {
            self.x: self.dataset.x_test,
            self.y: self.dataset.y_test
        }
        acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'test accuracy = {acc:.4f}')

    def create_weights(self, shape:list, stddev=0.05)->tf.Variable:
        return tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=stddev))

    def create_biases(self, size:int):
        return tf.Variable(tf.zeros([size]))

    def conv_layer(self, input, input_channels, filters, filter_size):
        weights = self.create_weights(shape=[filter_size, filter_size, input_channels, filters])
        biases = self.create_biases(filters)

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='VALID')
        layer += biases

        # layer = self.pool(layer=layer, ksize=[1,2,2,1], strides=[1,2,2,1])
        layer = tf.nn.relu(layer)

        return layer

    def pool(self, layer:tf.nn.conv2d, ksize:list, strides:list, padding='VALID'):
        return tf.nn.max_pool(layer, ksize=ksize, strides=strides, padding=padding)

    def flatten_layer(self, layer:tf.nn.conv2d):
        shape = layer.get_shape()
        features = shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, features])


        return layer

    def fc_layer(self, input, inputs, outputs, relu=True):
        weights = self.create_weights(shape=[inputs, outputs])
        biases = self.create_biases(outputs)

        layer = tf.matmul(input, weights)
        layer += biases

        if relu:
            layer = tf.nn.relu(layer)

        return layer


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
    epochs = 20

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    # gtsrb.display_one(gtsrb.x_train[0])

    start = datetime.now()
    cnn = CNN(gtsrb, num_epochs=20)
    end = datetime.now()
    print("Time taken to train the model on " + str(epochs) + " epochs is:", str(end - start))
