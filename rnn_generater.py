# to generate a sequence for 26 letters in the right order, and the aeiou 5 letter need to repeat 3 times to test the ablity of memorize
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import numpy as np
BATCH_SIZE = 16
_NUM_UNITS = 32
_NUM_LAYERS = 1
VOCAB_SIZE = 28

class Generator:
    def __init__(self):
        keep_prob = 0.5# dropout
        with tf.variable_scope('cell',initializer=tf.contrib.layers.xavier_initializer()):
            self.embedding = tf.get_variable(name = 'embedding',shape=[VOCAB_SIZE, _NUM_UNITS],trainable = True,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self._embed_ph = tf.placeholder(tf.float32, [VOCAB_SIZE, _NUM_UNITS])
            self._embed_init = self.embedding.assign(self._embed_ph)

            self.cell =  tf.nn.rnn_cell.BasicRNNCell(num_units=_NUM_UNITS)
#            self.initial_state = tf.placeholder(tf.float32, [_NUM_LAYERS * _NUM_UNITS * 2])
            h0 = self.cell.zero_state(BATCH_SIZE, np.float32)

            self.decoder_cell_drop = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=keep_prob)
            self.decoder_init_state = self.decoder_cell_drop.zero_state(BATCH_SIZE,dtype=tf.float32)

            self.decoder_lengths = tf.placeholder(tf.int32, [BATCH_SIZE])
            self.decoder_inputs = tf.placeholder(tf.int32, [BATCH_SIZE, None])
            self.outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = self.decoder_cell_drop,
                initial_state = self.decoder_init_state,#初始状态,h
                inputs = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs),#输入x
                sequence_length = self.decoder_lengths,
                dtype = tf.float32,scope = 'decoder')
        with tf.variable_scope('decoder'):
            self.softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, VOCAB_SIZE],initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_b = tf.get_variable('softmax_b', [VOCAB_SIZE],initializer=tf.contrib.layers.xavier_initializer())

        self.logits = tf.contrib.layers.fully_connected(self.outputs, VOCAB_SIZE, activation_fn=None,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                   biases_initializer=tf.zeros_initializer())
        self.probs = tf.nn.softmax(self.logits)
        self.targets = tf.placeholder(tf.int32, [BATCH_SIZE, None])  # 训练用
        self.labels = tf.one_hot(self.targets, depth=VOCAB_SIZE, dtype=tf.int32)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = self.probs,
                labels = self.labels)
        self.loss = tf.reduce_mean(loss)

        self.opt_op = tf.train.AdamOptimizer(0.01)
        self.train_op = self.opt_op.minimize(self.loss)




    def _init_vars(self, sess):
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

    def train(self):
        n_epoch = 100
        batch_size = BATCH_SIZE
        batch_num = 100
        length = 16
        embedding = np.random.random_sample([VOCAB_SIZE, _NUM_UNITS])
        with tf.Session() as sess:
            self._init_vars(sess)
            state = sess.run(self.decoder_init_state)
            for epoch in range(n_epoch):
                total_loss = 0
                temp_total_loss = 0
                for batch_i in range(batch_num):
                    batch = get_train_batch(batch_size,length - 2)
                    outputs, loss,_ = sess.run([self.decoder_final_state, self.loss, self.train_op], feed_dict={
                            self.decoder_init_state:state,
                            self.decoder_inputs: batch[:,:-1],
                            self.targets: batch[:,1:],
                            self.decoder_lengths:np.ones([batch_size])*length,
                            self._embed_ph: embedding})
                    total_loss+= loss
                    if batch_i % 10 == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f} range_loss = {:.3f}'.format(
                            epoch,
                            batch_i,
                            batch_num,
                            loss,
                            total_loss - temp_total_loss))
                        temp_total_loss = total_loss
                if epoch % 50 == 49:
                    outputs, loss, _ = sess.run([self.decoder_final_state, self.loss, self.train_op], feed_dict={
                        self.decoder_init_state: state,
                        self.decoder_inputs: batch[:, :-1],
                        self.targets: batch[:, 1:],
                        self.decoder_lengths: np.ones([batch_size]) * length,
                        self._embed_ph: embedding})
                    total_loss += loss

'''
temp_state, loss, temp_train_op,temp_prob,temp_labels,temp_logit,temp_outputs,temp_w,temp_b = sess.run([self.decoder_final_state, self.loss, self.train_op, self.probs,self.labels,self.logits,self.outputs,self.softmax_w,self.softmax_b], feed_dict={
    self.decoder_init_state: state,
    self.decoder_inputs: replaced_batch[:, :-1],
    self.targets: target_batch[:, 1:],
    self.decoder_lengths: length[batch_i],
    self._embed_ph: embedding})
'''




def get_train_batch(batch_size,length):
    ch_list = 'aaabcdeeefghiiijklmnooopqrstuuuvwxyzaaabcdeeefghiiijklmnooopqrstuuuvwxyzaaabcdeeefghiiijklmnooopqrstuuuvwxyzaaabcdeeefghiiijklmnooopqrstuuuvwxyz'
    #len(ch_list) = 144
    batch = []
    for i in range(batch_size):
        a = random.randint(0,144 - length)
        batch.append(ch_list[a:a+length])
    batch_2 = []
    for i in range(batch_size):
        batch_2.append(0)
        for j in range(length):
            batch_2.append(ord(batch[i][j]) - 95)
        batch_2.append(1)
    batch = np.array(batch_2).reshape([batch_size,length+2])
    return batch


if __name__ == '__main__':
    net = Generator()
    net.train()