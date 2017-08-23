""" 
 Linear regression in TensorFlow
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import utils

DATA_FILE = './fire_theft.xls'

## Prepare the graph

#1: read data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

#2: Create placeholders (type float32)
#     : input X (number of fire)
#     : label Y (number of theft)

#3: Create variables (initialized to 0)
#     : w (weight)
#     : b (bias)

#4: predict Y (number of theft) from the number of fire

#5: loss function: use the square error

#6: use gradient descent with learning rate of 0.001 to minimize loss


## Train the model

with tf.Session() as sess:
	#7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer())

	# Step 8: train the model
	for i in range(100): # run 100 epochs
		total_loss = 0
		for x, y in data:
			# Session runs optimizer: to minimize loss and to fetch the value of loss
			# The fetched value of loss should be stored in the variable l

			total_loss += l
		print("Epoch {0}: {1}".format(i, total_loss/n_samples))
	
	writer.close()

	#9: Output the values of w and b

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()