
import numpy as np
from wordcnt import wordcnt
from batch import getTrainBatch

# load word list
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

numFiles = wordcnt()
maxSeqLength = 250 # check with wordcnt.py

import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def doc2ids():
	ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
	fileCounter = 0
	for pf in positiveFiles:
	   with open(pf, "r") as f:
	       indexCounter = 0
	       line=f.readline()
	       cleanedLine = cleanSentences(line)
	       split = cleanedLine.split()
	       for word in split:
	           try:
	               ids[fileCounter][indexCounter] = wordsList.index(word)
	           except ValueError:
	               ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
	           indexCounter = indexCounter + 1
	           if indexCounter >= maxSeqLength:
	               break
	       fileCounter = fileCounter + 1 

	for nf in negativeFiles:
	   with open(nf, "r") as f:
	       indexCounter = 0
	       line=f.readline()
	       cleanedLine = cleanSentences(line)
	       split = cleanedLine.split()
	       for word in split:
	           try:
	               ids[fileCounter][indexCounter] = wordsList.index(word)
	           except ValueError:
	               ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
	           indexCounter = indexCounter + 1
	           if indexCounter >= maxSeqLength:
	               break
	       fileCounter = fileCounter + 1 
	#Pass into embedding function and see if it evaluates. 

	np.save('idsMatrix', ids)
	return ids

try:
	ids = np.load('idsMatrix.npy')	
	print('Loaded idsMatrix.npy\n')
except:
	ids = doc2ids()

batchSize = 512 #24
lstmUnits = 64
numClasses = 2
iterations = 25000
numDimensions = 50

import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
# value shape is [batch_size, max_time, output_size]

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
#last = tf.gather(value, int(value.get_shape()[0]) - 1)
last = value[-1]
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch(batchSize, maxSeqLength, ids);
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
   
   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

   #Save the network every 1000 training iterations
   if (i % 1000 == 0 or i == iterations-1):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)
writer.close()