'''
CNN to classify MNIST handwritten digits
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

# Read in MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("~/data/mnist", one_hot=True)

LOGDIR='mnist_result/1'

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 100

# Network Parameters
n_input = 784 # input image shape = 28*28 grey scale
n_classes = 10 # 10 classes (0-9 digits)
dropout = 0.75 # probability to keep units during dropout

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="label")
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Wrappers
def reshape(x, xdim, ydim):
    return tf.reshape(x, shape=[-1, xdim, ydim, 1])

def conv2d(x, W, b, stride=1, name="Conv"):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        #x = tf.nn.bias_add(x, b)
        tf.summary.histogram('Weight', W)
        tf.summary.histogram('Bias', b)
        act = tf.nn.relu(x + b)
        tf.summary.histogram('Activation', act)
        return act

def maxpool2d(x, size=2, stride=2, name="Pool"):
    with tf.name_scope(name):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    with tf.name_scope("model"):
        # Reshape input picture
        x = reshape(x, 28, 28)

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'], name="Conv1")
        print("Conv 1 = ", conv1)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, size=2, stride=2, name="Pool1")
        print("Conv 1 = ", conv1)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name="Conv2")
        print("Conv 2 = ", conv2)

        # Max Pooling (down-sampling) size=2, stride=2
        conv2 = maxpool2d(conv2, size=2, stride=2, name="Pool2")
        print("Conv 2 = ", conv2)

        with tf.name_scope("FC1"):
            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, dropout)

        with tf.name_scope("FC2"):
            # Output, class prediction
            out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
            return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="Filter1"),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="Filter2"),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name="FC1_W"),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name="FC2_W")
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name="Filter1_Bias"),
    'bc2': tf.Variable(tf.random_normal([64]), name="Filter2_Bias"),
    'bd1': tf.Variable(tf.random_normal([1024]), name="FC1_Bias"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="FC2_Bias")
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    tf.summary.scalar("Loss", loss)

# Evaluate model
with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

#tf.summary.image('Test Image', tf.reshape(mnist.test.images, [-1,28,28,1]), 10)

summ_merged = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            l, acc, s = sess.run([loss, accuracy, summ_merged], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            writer.add_summary(s, step)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(l) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256], keep_prob: 1.}))

    writer.close()
