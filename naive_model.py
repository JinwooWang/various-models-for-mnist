import tensorflow as tf
from tensorflow.contrib import layers

class CNN:
    def __init__(self, inputs, num_classes, keep_prob):
        self.inputs = inputs
        self.num_classes = num_classes
        self.keep_prob = keep_prob

    def model(self):
        regularizer = layers.l2_regularizer(scale = 0.01)
        x = tf.layers.conv2d(self.inputs, 32, 3, 1, "SAME",
                             kernel_initializer = layers.xavier_initializer_conv2d(),
                             kernel_regularizer = regularizer,
                             name = "conv1")
        x = tf.nn.relu(x, name = "relu1")
        x = tf.layers.average_pooling2d(x, 2, 2, "SAME", name = "pool1")
        x = tf.layers.dropout(x, self.keep_prob)

        x = tf.layers.conv2d(x, 64, 3, 1, "SAME",
                             kernel_initializer = layers.xavier_initializer_conv2d(),
                             kernel_regularizer = regularizer,
                             name = "conv2")
        x = tf.nn.relu(x, name = "relu2")
        x = tf.layers.average_pooling2d(x, 2, 2, "SAME", name = "pool2")
        x = tf.layers.dropout(x, self.keep_prob)

        x = tf.layers.conv2d(x, 128, 3, 1, "SAME",
                             kernel_initializer = layers.xavier_initializer_conv2d(),
                             kernel_regularizer = regularizer,
                             name = "conv3")
        x = tf.nn.relu(x, name = "relu3")
        x = tf.layers.average_pooling2d(x, 2, 2, "SAME", name = "pool3")


        x = tf.layers.conv2d(x, 256, 3, 1, "SAME",
                             kernel_initializer = layers.xavier_initializer_conv2d(),
                             kernel_regularizer = regularizer,
                             name = "conv4")
        x = tf.nn.relu(x, name = "relu4")
        x = tf.layers.average_pooling2d(x, 2, 2, "SAME", name = "pool4")

        x = tf.layers.flatten(x, name = "flatten")
        x = tf.layers.dense(x, 700, name = "dense1", kernel_regularizer = regularizer)
        x = tf.layers.dropout(x, self.keep_prob)

        x = tf.layers.dense(x, self.num_classes, name = "dense2")
        return x
