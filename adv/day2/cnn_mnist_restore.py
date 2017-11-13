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
SAVEDIR='mnist/model.ckpt'

# Parameters
learning_rate = 0.001
training_iters = 5000
batch_size = 128
display_step = 1

# Network Parameters
n_input = 784 # input image shape = 28*28 grey scale
n_classes = 10 # 10 classes (0-9 digits)
dropout = 0.75 # probability to keep units during dropout

# Launch the graph
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./mnist/model.ckpt.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./mnist'))
	#sess.run(tf.global_variables_initializer())
	# Calculate accuracy for 256 mnist test images
	
	graph  = tf.get_default_graph()
	accuracy = graph.get_tensor_by_name("accuracy:0")
	x = graph.get_tensor_by_name("x:0")
	y = graph.get_tensor_by_name("y:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")

	print("Testing Accuracy:", \
		sess.run(accuracy, feed_dict={x: mnist.test.images[:256], 
			y: mnist.test.labels[:256], keep_prob: 1.}))

