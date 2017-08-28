""" 
Softmax logistic regression to classify MNIST handwritten digits
MNIST dataset: yann.lecun.com/exdb/mnist/
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 5

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('~/data/mnist', one_hot=True) 

# Step 2: create placeholders
#     : MNIST image has 28*28 = 784 pixels  => 1x784 tensor
#     : 10 classes for digits 0 - 9 (one-hot vector)
X = tf.placeholder(tf.float32, [batch_size, 784], name='image') 
Y = tf.placeholder(tf.int32, [batch_size, 10], name='digit')

# Step 3: create weights and bias
# w: init. to random values with mean 0, stddev 0.01
# b: init. to 0
# shape of w: depends on the dim. of X and Y, so that Y = tf.matmul(X, w)
# shape of b: depends on Y
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# Step 4: build logit model
logits = tf.matmul(X, w) + b 

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

	start_time = time.time()

	sess.run(tf.global_variables_initializer())	

	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0
		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') 

	# test the model
	
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, axis=1), tf.argmax(Y, axis=1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
	
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy, feed_dict={X: X_batch, Y:Y_batch}) 
		total_correct_preds += accuracy_batch	
	
	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

writer.close()