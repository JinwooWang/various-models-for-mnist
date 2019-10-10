from tensorflow.contrib import layers
import tensorflow as tf

class ResNet50:
    def __init__(self, inputs, num_classes, training, momentum = 0.99):
        self.inputs = inputs
        self.num_classes = num_classes
        self.training = training
        self.momentum = momentum

    def model(self):
        filters1 = [64, 64, 256]
        filters2 = [128, 128, 512]
        filters3 = [256, 256, 1024]
        filters4 = [512, 512, 2048]
        with tf.variable_scope("stage1"):
            x = tf.layers.conv2d(self.inputs, 64, 7, 2, 'SAME',
                                 kernel_initializer = layers.xavier_initializer_conv2d(),
                                 name = "conv1")
            x = tf.layers.batch_normalization(x, momentum = self.momentum, training = self.training, name = "bn1")
            x = tf.nn.relu(x, name = "relu1")
            x = tf.layers.max_pooling2d(x, 3, 2, "SAME", name = "max_pool1")

        with tf.variable_scope("stage2"):
            x = self.conv_block(filters1, x, 3, 2, "stage2", "a")
            for i in range(2):
                x = self.identity_block(filters1, x, 3, "stage2", chr(ord("b") + i))

        with tf.variable_scope("stage3"):
            x = self.conv_block(filters2, x, 3, 2, "stage3", "a")
            for i in range(3):
                x = self.identity_block(filters2, x, 3, "stage3", chr(ord("b") + i))

        with tf.variable_scope("stage4"):
            x = self.conv_block(filters3, x, 3, 2, "stage4", "a")
            for i in range(5):
                x = self.identity_block(filters3, x, 3, "stage4", chr(ord("b") + i))

        with tf.variable_scope("stage5"):
            x = self.conv_block(filters4, x, 3, 2, "stage5", "a")
            for i in range(2):
                x = self.identity_block(filters4, x, 3, "stage5", chr(ord("b") + i))

        with tf.variable_scope("output"):
            pool_size = x.get_shape().as_list()[1]
            x = tf.layers.average_pooling2d(x, pool_size, 1, name = "avg_pool")
            x = tf.layers.flatten(x, name = "flatten")
            x = tf.layers.dense(x, self.num_classes, name = "dense")

        return x

    def conv_block(self, filters, inputs, kernel_size, stride, stage_name, block_name):
        filter1, filter2, filter3 = filters
        base_name = stage_name + "_" + block_name
        with tf.variable_scope(base_name):
            with tf.variable_scope(base_name + "_layer1"):
                x = tf.layers.conv2d(inputs, filter1, 1, stride, "VALID",
                                     kernel_initializer = layers.xavier_initializer_conv2d(),
                                     name = base_name + "_conv1")
                x = tf.layers.batch_normalization(x, momentum = self.momentum,
                                                  training = self.training,
                                                  name = base_name + "_bn1")
                x = tf.nn.relu(x, name = base_name + "_relu1")

            with tf.variable_scope(base_name + "_layer2"):
                x = tf.layers.conv2d(x, filter2, kernel_size, 1, "SAME",
                                     kernel_initializer = layers.xavier_initializer_conv2d(),
                                     name = base_name + "_conv2")
                x = tf.layers.batch_normalization(x, momentum = self.momentum,
                                                  training = self.training,
                                                  name = base_name + "_bn2")
                x = tf.nn.relu(x, name = base_name + "_relu2")

            with tf.variable_scope(base_name + "_layer3"):
                x = tf.layers.conv2d(x, filter3, 1, 1, "VALID",
                                     kernel_initializer = layers.xavier_initializer_conv2d(),
                                     name = base_name + "_conv3")
                x = tf.layers.batch_normalization(x, momentum = self.momentum,
                                                  training = self.training,
                                                  name = base_name + "_bn3")
            with tf.variable_scope(base_name + "_shortcut"):
                shortcut = tf.layers.conv2d(inputs, filter3, 1, stride, "VALID",
                                     kernel_initializer = layers.xavier_initializer_conv2d(),
                                     name = base_name + "_sc_conv1")
                shortcut = tf.layers.batch_normalization(shortcut, momentum = self.momentum,
                                                  training = self.training,
                                                  name = base_name + "_sc_bn1")

            with tf.variable_scope(base_name + "_fusion"):
                x = tf.add(x, shortcut, name = base_name + "_add")
                x = tf.nn.relu(x, name = base_name + "_relu4")
            return x

    def identity_block(self, filters, inputs, kernel_size, stage_name, block_name):
        filter1, filter2, filter3 = filters
        base_name = stage_name + block_name
        with tf.variable_scope(base_name):
            with tf.variable_scope(base_name + "_layer1"):
                x = tf.layers.conv2d(inputs, filter1, 1, 1, "VALID",
                                     kernel_initializer = layers.xavier_initializer_conv2d(),
                                     name = base_name + "_conv1")
                x = tf.layers.batch_normalization(x, momentum = self.momentum,
                                                  training = self.training,
                                                  name = base_name + "_bn1")
                x = tf.nn.relu(x, name = base_name + "_relu1")

            with tf.variable_scope(base_name + "_layer2"):
                x = tf.layers.conv2d(x, filter2, kernel_size, 1, "SAME",
                                     kernel_initializer = layers.xavier_initializer_conv2d(),
                                     name = base_name + "_conv2")
                x = tf.layers.batch_normalization(x, momentum = self.momentum,
                                                  training = self.training,
                                                  name = base_name + "_bn2")
                x = tf.nn.relu(x, name = base_name + "_relu2")

            with tf.variable_scope(base_name + "_layer3"):
                x = tf.layers.conv2d(x, filter3, 1, 1, "VALID",
                                     kernel_initializer = layers.xavier_initializer_conv2d(),
                                     name = base_name + "_conv3")
                x = tf.layers.batch_normalization(x, momentum = self.momentum,
                                              training = self.training,
                                              name = base_name + "_bn3")

            with tf.variable_scope(base_name + "_fusion"):
                x = tf.add(x, inputs, name = base_name + "_add")
                x = tf.nn.relu(x, name = base_name + "_relu3")
            return x
