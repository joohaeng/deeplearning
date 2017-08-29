""" 
    Character-level generative language model.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import utils
import time


DATA_PATH = 'data/arvix_abstracts.txt'
HIDDEN_SIZE = 200
BATCH_SIZE = 64
NUM_STEPS = 50
DISP_STEP = 40
#TEMPRATURE = 0.7
LEARNING_RATE = 0.003
LEN_GENERATED = 300

def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window=NUM_STEPS, overlap=NUM_STEPS//2):
    for text in open(filename):
        text = vocab_encode(text, vocab)
        for start in range(0, len(text) - window, overlap):
            chunk = text[start: start + window]
            chunk += [0] * (window - len(chunk))
            yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

def create_rnn(seq, hidden_size=HIDDEN_SIZE):
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)  
    #in_state = tf.placeholder_with_default(
    #        cell.zero_state(tf.shape(seq)[0], tf.float32), [None, hidden_size])

    in_state = cell.zero_state(tf.shape(seq)[0], tf.float32)
    #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.75)

    #print(in_state)
    # this line to calculate the real length of seq
    # all seq are padded to be of the same length which is NUM_STEPS
    length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state)

    return output, in_state, out_state

def create_model(seq, vocab, hidden=HIDDEN_SIZE):
    seq = tf.one_hot(seq, len(vocab))
    output, in_state, out_state = create_rnn(seq, hidden) # output shape = [batch_size, max_time, cell.output_size]
    # fully_connected is syntactic sugar for tf.matmul(w, output) + b
    # it will create w and b for us
    logits = tf.contrib.layers.fully_connected(output, len(vocab), activation_fn=None)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=seq[:, 1:]))
    sample = tf.multinomial(logits[:, -1], num_samples=1)[:, 0] 
    return loss, sample, in_state, out_state

def online_inference(sess, vocab, seq, sample, in_state, out_state, seed='T'):
    """ Generate sequence one character at a time, based on the previous character
    """
    sentence = seed
    state = None
    for _ in range(LEN_GENERATED):
        batch = [vocab_encode(sentence[-1], vocab)]
        feed = {seq: batch}
        # for the first decoder step, the state is None
        if state is not None:
            feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += vocab_decode(index, vocab)
    print(sentence)

vocab = (
        " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "\\^_abcdefghijklmnopqrstuvwxyz{|}")
seq = tf.placeholder(tf.int32, [None, None])
loss, sample, in_state, out_state = create_model(seq, vocab)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
  
    iteration = 0
    for batch in read_batch(read_data(DATA_PATH, vocab)):

        batch_loss, _ = sess.run([loss, optimizer], {seq: batch})

        if (iteration + 1) % DISP_STEP == 0:
            print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
            online_inference(sess, vocab, seq, sample, in_state, out_state)
            start = time.time()
        iteration += 1
