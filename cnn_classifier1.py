import numpy as np
import tensorflow as tf
import os
import warnings
from data_loader import ImageDataset
from datetime import datetime
import matplotlib.pyplot as plt


class CNN():

    tf_sess = None
    model = None
    dataset = None
    batch_size = 128
    repeat_size = 5
    shuffle = 128
    learning_rate = 0.001

    def __init__(self, dataset:ImageDataset, load_dataset=True, num_epochs=100, learning_rate=0.001,
                 enable_session=False, dynamic_lr=True, shape=None, num_classes=None):

        if enable_session:
            self.tf_sess = tf.Session()

        if load_dataset:
            self.dataset = dataset
        # self.setup_batch_iterator()
        self.build_model(epochs=num_epochs,
                         learning_rate=learning_rate,
                         enable_dynamic_lr=dynamic_lr,
                         dataset_loaded=load_dataset,
                         shape=shape,
                         num_classes=num_classes)

    def build_model(self, epochs=50, learning_rate=0.001, enable_dynamic_lr=True, dataset_loaded=True, shape=None, num_classes=None):
        print("Building model...")
        if dataset_loaded:
            self.x = tf.placeholder(tf.float32, [None] + self.dataset.shape)
        else:
            self.x = tf.placeholder(tf.float32, [None] + shape)

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
        print()

        # Convert train data labels to one hot encoding to feed into softmax function
        if dataset_loaded:
            y_to_one_hot = tf.one_hot(self.y, self.dataset.num_classes)
        else:
            y_to_one_hot = tf.one_hot(self.y, num_classes)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_to_one_hot)

        self.loss = tf.reduce_mean(cross_entropy)
        # print("Loss: ", self.loss)


        correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_to_one_hot, axis=1))
        # correct =
        # print(correct)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # print("Accuracy", self.accuracy)
        self.prediction = tf.argmax(logits, axis=1)
        # print("Prediction", self.prediction)

        # Get starting learning rate
        if enable_dynamic_lr:
            self.learning_rate = self.get_optimal_learning_rate(epochs=epochs, learning_rate=learning_rate, plot_charts=False)
        else:
            self.learning_rate = learning_rate

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # print("Optimizer:", self.optimizer)

    def train_model(self, epochs:int, limit=6):
        print("Training model...")
        self.tf_sess.run(tf.global_variables_initializer())

        best, no_change, total_loss, total_acc = 0, 0, 0, 0

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
                        self.x: self.dataset.preprocess(bx),#.reshape((-1, 32, 32, 3)),
                        self.y: by#.reshape((-1))
                    }
                    self.tf_sess.run(self.optimizer, feed_dict=feed_dict)
                    loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    total += acc * len(by)
                    total_loss += loss * len(by)
                    total_acc += acc * len(by)
                    # print(total / len(self.dataset.y_train))
            except(tf.errors.OutOfRangeError):
                pass


            feed_dict = {
                self.x: self.dataset.preprocess_normalize_only(self.dataset.x_valid),
                self.y: self.dataset.y_valid
            }

            vloss, vacc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            print(f'epoch {epoch + 1}: loss = {vloss:.4f}, '
                  f'training accuracy = {total / len(self.dataset.y_train):.4f}, '
                  f'validation accuracy = {vacc:.4f}, '
                  f'learning rate = {self.learning_rate:.10f}')

            # Early stopping
            if vacc > best:
                best = vacc
                no_change = 0
            else:
                no_change += 1

            if no_change >= limit:
                print("Early stopping...")
                break

        feed_dict = {
            self.x: self.dataset.preprocess_normalize_only(self.dataset.x_test),
            self.y: self.dataset.y_test
        }
        acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'test accuracy = {acc:.4f}')

    def get_optimal_learning_rate(self, epochs=50, learning_rate=1e-5, plot_charts=False):
        # if self.tf_sess is not None and self.tf_sess._closed == False:
        #     print("Restarting session for learning rate...")
        #     self.tf_sess.close()
        #     self.tf_sess = tf.Session()
        # else:
        #     print("Creating new session for learning rate...")
        #     self.tf_sess = tf.Session()

        print("Finding optimal learning rate...")
        self.tf_sess.run(tf.global_variables_initializer())
        rates = list()
        t_loss = list()
        t_acc = list()

        self.tf_sess.run(self.dataset.train_init)
        for i in range(epochs):
            # Store learning rate in a tf variable and update it
            # g_step = tf.Variable(0, trainable=False)
            # lr = tf.train.exponential_decay(learning_rate, g_step, 100000, 0.96, staircase=True)

            learning_rate *= 1.1
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

            bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])
            feed_dict = {
                self.x: bx,
                self.y: by
            }

            self.tf_sess.run(optimizer, feed_dict=feed_dict)
            loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            if np.isnan(loss):
                loss = np.nan_to_num(loss)
            rates.append(learning_rate)
            t_loss.append(loss)
            t_acc.append(acc)

            print(f'epoch {i + 1}: learning rate = {learning_rate:.10f}, loss = {loss:.10f}')
        if plot_charts:
            iters = np.arange(len(rates))
            plt.title("Learning Rate (log) vs. Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Learning Rate")
            plt.plot(iters, rates, 'b')
            plt.show()

            plt.plot(rates, t_loss, 'b')
            plt.title("Loss vs. Learning Rate (log)")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.show()

        # Calculate the learning rate based on the biggest derivative betweeen the loss and learning rate
        dydx = list(np.divide(np.diff(t_loss), np.diff(rates)))
        start = rates[dydx.index(max(dydx))]
        print("Chosen start learning rate:", start)
        print()
        # self.tf_sess.close()
        return start

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

    epochs = 50
    learning_rate = 1e-3
    enable_session = True
    dynamic_lr = True
    shape = (32,32,3)
    num_classes = 43

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    print()

    # gtsrb.display_one(gtsrb.x_train[0])

    start = datetime.now()
    cnn = CNN(dataset=gtsrb,
              num_epochs=epochs,
              learning_rate=learning_rate,
              enable_session=enable_session,
              dynamic_lr=dynamic_lr,
              shape=shape,
              num_classes=num_classes)
    if enable_session:
        cnn.train_model(epochs=epochs, limit=8)
        end = datetime.now()
        print("Time taken to build and train the model on " + str(epochs) + " epochs is:", str(end - start))
        cnn.tf_sess.close()
    else:
        end = datetime.now()
        print("Time taken to build the model on " + str(epochs) + " epochs is:", str(end - start))