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

        self.q = self.build_Q_net("Qnet", trainable=True)
        self.q_next = self.build_Qnext_net("Qnext", trainable=False)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def read_data(self):


    def build_Q_net(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tf_s, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            l3 = tf.layers.dense(l2, 1, trainable=trainable)
            return l3

    def build_Qnext_net(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tf_s_, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            l3 = tf.layers.dense(l2, 1, trainable=trainable)
            return l3

    def learn(self):
        s_list, a_list, snext_list, r_list, done_list = self.read_batch()
        q_eval = []
        q_target = []

        a_indices = tf.stack([tf.range(tf.shape(self.tf_a)[0], dtype=tf.int32), self.tf_a], axis=1)
        q_eval = tf.gather_nd(params=self.q, indices=a_indices)
        for i in range(self.BATCH_SIZE):
            if done_list[i] == 0:
                q_target.append(r_list[i] + self.GAMMA * np.max(self.q_next.eval(
                    session=self.sess, feed_dict={self.tf_s_:snext_list})))
            else:
                q_target.append(r_list[i])

        td_error = tf.reduce_mean(tf.squared_difference(q_target, q_eval))
        train_op = tf.train.AdamOptimizer(self.LR).minimize(td_error)
        self.sess.run(train_op,{self.tf_s:s_list,
            self.a_list:a_list, self.tf_s_:snext_list,self.tf_a:a_list})


    def read_batch(self):
        s_list = []
        a_list = []
        snext_list = []
        r_list = []
        done_list = []

        for i in range(self.BATCH_SIZE):
            index = random.randint(0, len(self.s_list) - 1)
            s_list.append(self.s_list[index])
            a_list.append(self.a_list[index])
            snext_list.append(self.snext_list[index])
            r_list.append(self.r_list[index])
            done_list.append(self.done_list[index])

        return s_list, a_list, snext_list, r_list, done_list
