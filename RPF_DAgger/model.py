import os
import tensorflow as tf
import cv2
import joblib
from ops import *
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt


batch_size = 4
GRU_size = 512
action_dim = 4
learning_rate = 1e-4
demo_length = 30
max_follow_length = 50
memory_dim = 512
EPOCH = 500
img_size = 256
feature_dim = 512

def FCN_2layer(scope, x, hidden_ch=1024, in_ch=512 * 8 * 8 + action_dim, out_ch=1, reuse=tf.AUTO_REUSE,
               activation='tanh'):  # tf.AUTO_REUSE):

    with tf.variable_scope(scope, reuse=reuse):
        init = tf.contrib.layers.xavier_initializer(uniform=True)
        W1 = tf.get_variable('W1', shape=[in_ch, hidden_ch], initializer=init)
        b1 = tf.get_variable('b1', [hidden_ch], initializer=tf.constant_initializer(0.0))
        x = tf.add(tf.matmul(x, W1), b1)

        W2 = tf.get_variable('W2', shape=[x.get_shape()[-1], out_ch], initializer=init)
        b2 = tf.get_variable('b2', [out_ch], initializer=tf.constant_initializer(0.0))
        x = tf.add(tf.matmul(x, W2), b2)

        if activation == 'tanh':
            return tf.nn.tanh(x)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif activation == 'softmax':
            return tf.nn.softmax(x)
        elif activation == 'none':
            return x
        # NO ACTIVATION OPTION FOR THE SOFTMAX CROSS ENTROPY! soft max 로 바꾸기


def CNN(scope, x, x_dim=3, hidden_ch=feature_dim * 2, output_ch=feature_dim, num_layer=6, start_ch=16,
        reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        init = tf.contrib.layers.xavier_initializer(uniform=True)

        for i in range(1, num_layer + 1):
            in_dim = x_dim if i == 1 else x.get_shape()[-1]
            ch = start_ch if i == 1 else x.get_shape()[-1] * 2
            W = tf.get_variable('W{}'.format(i), [4, 4, in_dim, ch], initializer=init)
            b = tf.get_variable('b{}'.format(i), [ch], initializer=tf.constant_initializer(0.0))
            # x = lrelu(batch_normal(max_pool(conv2d(x, W, b)),scope='phi{}'.format(i))) # 128
            x = lrelu(max_pool(conv2d(x, W, b)))  # 128 # Add two fc layer (fc - relu - fc)

        x = tf.layers.flatten(x)

        out_size = img_size / 2 ** num_layer
        out_ch = start_ch * (2 ** (num_layer - 1))
        # print(out_size, out_ch)

        W1 = tf.get_variable('W{}'.format(num_layer + 1), shape=[out_ch * (out_size * out_size), hidden_ch],
                             initializer=init)
        b1 = tf.get_variable('b{}'.format(num_layer + 1), [hidden_ch], initializer=tf.constant_initializer(0.0))
        x = lrelu(tf.add(tf.matmul(x, W1), b1))

        W2 = tf.get_variable('W{}'.format(num_layer + 2), shape=[x.get_shape()[-1], output_ch], initializer=init)
        b2 = tf.get_variable('b{}'.format(num_layer + 2), [output_ch], initializer=tf.constant_initializer(0.0))
        x = tf.add(tf.matmul(x, W2), b2)

        return x


class RPF():
    def __init__(self):
        self.batch_size = batch_size
        self.record = {'actions_t': [], 'eta': [], 'h_ts': [], 'attention_t': [], 'inp_feats_t': [], 'mu_t': []}
        self.follow_length = max_follow_length
        self.GRU_size = GRU_size
        self.demo_length = demo_length
        self.img_size = 256
        self.action_dim = action_dim

        self.curr_h_t = None
        self.curr_eta = None
        self.curr_memories = None
        self.sess = None

    def build_network_for_train(self, demo=None, placeholder=False):
        
        if placeholder:

            self.demo_seqs_list = tf.placeholder(tf.float32, [None, self.demo_length, self.img_size, self.img_size, 3])
            self.demo_acts_list = tf.placeholder(tf.float32,[None, self.demo_length, self.action_dim])
            self.inp_rgb_list = tf.placeholder(tf.float32, [None,  self.follow_length, self.img_size, self.img_size, 3])
            self.inp_act_list = tf.placeholder(tf.float32, [None, self.follow_length,  self.action_dim])
            self.act_loss_mask_list = tf.placeholder(tf.float32,[None,self.follow_length, self.action_dim])
        else:
            [self.demo_seqs_list, self.demo_acts_list, self.inp_rgb_list, self.inp_act_list, self.act_loss_mask_list] = demo

        
        demo_seqs_list = tf.transpose(self.demo_seqs_list, [1, 0, 2, 3, 4])
        demo_acts_list = tf.transpose(self.demo_acts_list, [1, 0, 2])
        inp_rgb_list = tf.transpose(self.inp_rgb_list, [1, 0, 2, 3, 4])
        inp_act_list = tf.transpose(self.inp_act_list, [1, 0, 2])
        act_loss_mask_list = tf.transpose(self.act_loss_mask_list, [1, 0, 2])

        memories_list = []
        for t in range(self.demo_length):
            demo_rgb_t = demo_seqs_list[t]  # B X img X img X 3
            demo_act_t = demo_acts_list[t]  # B X action_dim
            demo_rgb_feats_t = CNN('img_feature', demo_rgb_t)
            memories_inp = tf.concat([demo_rgb_feats_t, demo_act_t], -1)
            memories_t = FCN_2layer('memory', memories_inp, in_ch=feature_dim + action_dim, out_ch=memory_dim)
            # B X Features
            memories_list.append(memories_t)

        # memories_list = tf.concat(memories_list, 0) # demo_length X B X Features
        memories_list = tf.transpose(memories_list, [1, 0, 2])  # B X demo_length X Features
        eta = tf.ones([self.batch_size, 1])

        self.GRU_cell = tf.nn.rnn_cell.GRUCell(GRU_size, activation=tf.tanh)#, name='gru_cell', reuse=tf.AUTO_REUSE)
        h_t = self.GRU_cell.zero_state(self.batch_size, tf.float32)
        self.loss = 0
        for t in range(self.follow_length):
            self.record['h_ts'].append(h_t)
            self.record['eta'].append(eta)

            attention_t = tf.stack([tf.exp(-abs(eta - float(j))) for j in range(demo_length)], 1)
            # B X demo_length X 1
            self.record['attention_t'].append(attention_t)

            mu_t = tf.reduce_sum(tf.multiply(memories_list, attention_t), 1)  # B X Features
            self.record['mu_t'].append(mu_t)

            inp_rgb_t = inp_rgb_list[t]  # B X img X img X 3
            inp_act_t = inp_act_list[t]  # B X action_dim
            inp_feats_t = CNN('img_feature', inp_rgb_t)  # B X Features
            self.record['inp_feats_t'].append(inp_feats_t)

            gru_inp = tf.reshape(tf.concat([inp_feats_t, mu_t], -1)
                                 , [self.batch_size, memory_dim + feature_dim])
            h_t = self.GRU_cell(gru_inp, h_t)
            h_t = h_t[0]
            h_t = FCN_2layer('hidden_state_encode', h_t, in_ch=GRU_size, hidden_ch=GRU_size, out_ch=GRU_size,
                             activation='none')
            # B X GRU_size
            eta = eta + (1 + FCN_2layer('pointer_increment', h_t, in_ch=GRU_size, hidden_ch=512, out_ch=1,
                                        activation='tanh'))
            eta = tf.math.minimum(eta, 29)

            # B X 1
            pred_action = FCN_2layer('action', h_t, in_ch=GRU_size, hidden_ch=512, out_ch=action_dim, activation='none')
            self.record['actions_t'].append(pred_action)

            act_loss_mask = tf.reshape(act_loss_mask_list[t], [self.batch_size, action_dim])
            curr_loss = tf.losses.softmax_cross_entropy(inp_act_t, tf.multiply(pred_action, act_loss_mask))
            self.loss += curr_loss / self.follow_length

        self.opti_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def build_network_for_running(self, placeholder=False):
        # run this func after the basic build_network()

        self.run_demo_seqs_list = tf.placeholder(tf.float32, [None, self.demo_length, self.img_size, self.img_size, 3])
        self.run_demo_acts_list = tf.placeholder(tf.float32,[None, self.demo_length, self.action_dim])
        self.run_inp_rgb = tf.placeholder(tf.float32, [None,  self.img_size, self.img_size, 3])
        self.run_inp_act = tf.placeholder(tf.float32, [None, self.action_dim])
        self.run_h_t = tf.placeholder(tf.float32, [None,self.GRU_size])
        self.run_memories = tf.placeholder(tf.float32,[None,self.demo_length, memory_dim])
        self.run_eta = tf.placeholder(tf.float32,[None, 1])
        
        demo_seqs_list = tf.transpose(self.run_demo_seqs_list, [1, 0, 2, 3, 4])
        demo_acts_list = tf.transpose(self.run_demo_acts_list, [1, 0, 2])

        ###### self.run_demo_seqs_list, self.run_demo_acts_list
        memories_list = []
        for t in range(self.demo_length):
            demo_rgb_t = demo_seqs_list[t]  # B X img X img X 3
            demo_act_t = demo_acts_list[t]  # B X action_dim
            demo_rgb_feats_t = CNN('img_feature', demo_rgb_t)
            memories_inp = tf.concat([demo_rgb_feats_t, demo_act_t], -1)
            memories_t = FCN_2layer('memory', memories_inp, in_ch=feature_dim + action_dim, out_ch=memory_dim)
            # B X Features
            memories_list.append(memories_t)

        self.get_memories = tf.transpose(memories_list, [1, 0, 2])


        ##### self.run_inp_rgb, self.run_h_t, self.run_memories, self.run_eta
        inp_feats_t = CNN('img_feature', self.run_inp_rgb)  # B X Features

        attention_t = tf.stack([tf.exp(-abs(self.run_eta - float(j))) for j in range(demo_length)], 1)
        mu_t = tf.reduce_sum(tf.multiply(self.run_memories, attention_t), 1)  # B X Features

        self.GRU_cell = tf.nn.rnn_cell.GRUCell(GRU_size, activation=tf.tanh)#, name='gru_cell', reuse=tf.AUTO_REUSE)
        gru_inp = tf.reshape(tf.concat([inp_feats_t, mu_t], -1)
                             , [self.batch_size, memory_dim + feature_dim])
        h_t = self.GRU_cell(gru_inp, self.run_h_t)
        h_t = h_t[0]
        #h_t = FCN_2layer('hidden_state_encode', h_t, in_ch=GRU_size, hidden_ch=GRU_size, out_ch=GRU_size, activation='none')

        self.get_h_t = FCN_2layer('hidden_state_encode', h_t, in_ch=GRU_size, hidden_ch=GRU_size, out_ch=GRU_size, activation='none')

        eta = self.run_eta + (1 + FCN_2layer('pointer_increment', h_t, in_ch=GRU_size, hidden_ch=512, out_ch=1,
                                    activation='tanh'))
        self.get_eta = tf.math.minimum(eta, 29)
        self.get_action = tf.nn.softmax(FCN_2layer('action', h_t, in_ch=GRU_size, hidden_ch=512, out_ch=action_dim, activation='none'))
        self.get_rnn_output = [self.get_h_t, self.get_eta, self.get_action]



    def reset(self):
        self.curr_h_t = np.zeros([self.batch_size, self.GRU_size])
        self.curr_eta = np.ones([self.batch_size, 1])
        self.curr_memories = None

    def encode_memory(self,demo):
        self.curr_memories = self.sess.run(self.get_memories,feed_dict={self.run_demo_seqs_list:demo[0], self.run_demo_acts_list:demo[1]})
        print('Memory based on demonstration encoded - saved in RPF')
        return self.curr_memories

    def predict_action(self, curr_inp):
        feed_dict = {self.run_inp_rgb: curr_inp, self.run_h_t: self.curr_h_t, self.run_memories:self.curr_memories, self.run_eta:self.curr_eta}
        [self.curr_h_t, self.curr_eta, action] = self.sess.run(self.get_rnn_output, feed_dict=feed_dict)
        if np.random.rand() < 0.01 : print('eta : ', self.curr_eta, 'action ', action)

        return np.argmax(action,axis = -1)

