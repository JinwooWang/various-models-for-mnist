import tensorflow as tf
from resnet import ResNet50
from naive_model import CNN
from tensorflow.examples.tutorials.mnist import input_data

resnet_model_directory = "./resnet_models/"
resnet_log_directory = "reslog"
cnn_model_directory = "./cnn_models/"
cnn_log_directory = "cnnlog"

mnist = input_data.read_data_sets("./MNIST_data", one_hot = True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape([-1, 28, 28, 1])
teX = teX.reshape([-1, 28, 28, 1])

def trainResnet(checkpoint = None):
    X = tf.placeholder("float", [None, 28, 28, 1], "X")
    Y = tf.placeholder("float", [None, 10], "Y")

    resnet = ResNet50(X, 10, True)
    output_Y = resnet.model()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_Y, labels = Y))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    pred = tf.argmax(output_Y, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(Y, 1)), "float"))
    print(tf.trainable_variables())
    summary_loss = tf.summary.scalar("loss", loss)
    saver = tf.train.Saver(max_to_keep = 5)

    epochs = 5
    batch_size = 50
    iteration = 0

    merged_summary_op = tf.summary.merge_all()

    with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
        sess.run(tf.initialize_all_variables())
        if checkpoint:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        summary_writer = tf.summary.FileWriter(resnet_log_directory, tf.get_default_graph())
        for e in range(epochs):
            for i in range(len(trX) // batch_size + 1):
                iteration += 1
                batch = mnist.train.next_batch(batch_size)
                _, accuracy, summary = sess.run([train_op, acc, merged_summary_op], \
                                                 feed_dict = {X:batch[0].reshape([-1, 28, 28, 1]), Y:batch[1]})
                summary_writer.add_summary(summary, iteration)
                print(accuracy)
            saver.save(sess, save_path = resnet_model_directory + str(e) + ".ckpt")

def trainCNN(checkpoint = None):
    X = tf.placeholder("float", [None, 28, 28, 1], "X")
    Y = tf.placeholder("float", [None, 10], "Y")
    keep_prob = tf.placeholder("float")

    cnn = CNN(X, 10, keep_prob)
    output_Y = cnn.model()

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.96, staircase = True)
    l2_loss = tf.losses.get_regularization_loss()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_Y, labels = Y))
    loss += l2_loss
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_Y, 1), tf.argmax(Y, 1)), "float"))

    summary_lr = tf.summary.scalar("learning_rate", learning_rate)
    summary_loss = tf.summary.scalar("loss", loss)
    saver = tf.train.Saver(max_to_keep = 5)
    merged_summary_op = tf.summary.merge_all()

    epochs = 5
    batch_size = 50
    iteration = 0

    with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
        sess.run(tf.initialize_all_variables())
        if checkpoint:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        summary_writer = tf.summary.FileWriter(cnn_log_directory, tf.get_default_graph())
        for e in range(epochs):
            for i in range(len(trX) // batch_size + 1):
                iteration += 1
                batch = mnist.train.next_batch(batch_size)
                _, accuracy, summary = sess.run([train_op, acc, merged_summary_op], \
                                                feed_dict = {X:batch[0].reshape(-1, 28, 28, 1), Y:batch[1], keep_prob:0.5})
                summary_writer.add_summary(summary, iteration)
            saver.save(sess, save_path = cnn_model_directory + str(e) + ".ckpt")
            print(accuracy)



def test_resnet(checkpoint = resnet_model_directory):
    X = tf.placeholder("float", [None, 28, 28, 1], "X")
    Y = tf.placeholder("float", [None, 10], "Y")

    resnet = ResNet50(X, 10, False)
    output_Y = resnet.model()
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_Y, 1), tf.argmax(Y, 1)), "float"))
    saver = tf.train.Saver()
    print(tf.trainable_variables())

    with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        accuracy = sess.run([acc], feed_dict = {X:teX, Y:teY})
    print(accuracy)

def test_cnn(checkpoint = cnn_model_directory):
    X = tf.placeholder("float", [None, 28, 28, 1], "X")
    Y = tf.placeholder("float", [None, 10], "Y")
    keep_prob = tf.placeholder("float")

    cnn = CNN(X, 10, keep_prob)
    output_Y = cnn.model()
    print(tf.trainable_variables())
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_Y, 1), tf.argmax(Y, 1)), "float"))
    saver = tf.train.Saver()

    with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        accuracy = sess.run([acc], feed_dict = {X:teX, Y:teY, keep_prob:1.0})
    print(accuracy)

if __name__ == "__main__":
    #trainResnet()
    #test_resnet()
    #trainCNN()
    test_cnn()
