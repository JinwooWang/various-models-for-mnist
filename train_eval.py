import tensorflow as tf
from resnet import ResNet50
from tensorflow.examples.tutorials.mnist import input_data
model_directory = "./tfmodels/"
mnist = input_data.read_data_sets("./MNIST_data", one_hot = True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.train.labels
trX = trX.reshape([-1, 28, 28, 1])
teX = teX.reshape([-1, 28, 28, 1])

def train(checkpoint = None):
    X = tf.placeholder("float", [None, 28, 28, 1], "X")
    Y = tf.placeholder("float", [None, 10], "Y")

    resnet = ResNet50(X, 10, True)
    output_Y = resnet.model()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_Y, labels = Y))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    pred = tf.argmax(output_Y, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(Y, 1)), "float"))

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
        summary_writer = tf.summary.FileWriter('tflogs', tf.get_default_graph())
        for e in range(epochs):
            for i in range(len(trX) // batch_size + 1):
                iteration += 1
                batch = mnist.train.next_batch(batch_size)
                _, accuracy, summary = sess.run([train_op, acc, merged_summary_op], \
                                                 feed_dict = {X:batch[0].reshape([-1, 28, 28, 1]), Y:batch[1]})
                summary_writer.add_summary(summary, iteration)
                print(accuracy)
            saver.save(sess, save_path = model_directory + str(e) + ".ckpt")

def test(checkpoint = model_directory):
    X = tf.placeholder("float", [None, 28, 28, 1], "X")
    Y = tf.placeholder("float", [None, 10], "Y")

    resnet = ResNet50(X, 10, False)
    output_Y = resnet.model()
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_Y, 1), tf.argmax(Y, 1)), "float"))
    saver = tf.train.Saver()

    with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        accuracy = sess.run([acc], feed_dict = {X:teX, Y:teY})
    print(accuracy)


if __name__ == "__main__":
    #train()
    #test()
