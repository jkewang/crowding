import tensorflow as tf
import numpy as np
import random
import math

class DQFD(object):
    def __init__(self):
        #hyper params
        self.LR = 1e-3
        self.GAMMA = 0.9
        self.BATCH_SIZE = 32

        self.s_list = []
        self.a_list = []
        self.snext_list = []
        self.r_list = []
        self.done_list = []

        self.N_State = 292
        self.N_Action = 5
        self.tf_s = tf.placeholder(tf.float32, [None, self.N_State])
        self.tf_s_ = tf.placeholder(tf.float32, [None, self.N_State])
        self.tf_a = tf.placeholder(tf.float32, [None,self.N_Action])
        self.tf_r = tf.placeholder(tf.float32, [None,])

        self.q = self.build_Qnet("Qnet", trainable=True)
        self.q_next = self.build_Qnet("Qnext", trainable=False)

        self.td_error = tf.reduce_mean(tf.squared_difference(self.q ,self.q_next))
        self.train_op = tf.tf.train.AdamOptimizer(self.LR).minimize(self.td_error)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def read_data(self):


    def build_Qnet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tf_s, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            l3 = tf.layers.dense(l2, 1, trainable=trainable)
            return l3

    def learn(self):

    def read_batch(self):

