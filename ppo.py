import numpy as np
import random
import math
import tensorflow as tf

GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 1000
S_DIM, A_DIM = 292, 5

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            l2 = tf.layers.dense(l1,100,tf.nn.relu)
            l3 = tf.layers.dense(l2,100,tf.nn.relu)
            self.v = tf.layers.dense(l3, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
            tf.summary.scalar('closs', self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
                pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
                oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
                ratio = pi_prob / oldpi_prob
                surr = ratio * self.tfadv
                self.aloss = -tf.reduce_mean(tf.minimum(surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
                tf.summary.scalar('aloss', self.aloss)

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        self.writer = tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        #adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        # update actor
        a = np.hstack(a)
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        print("aloss",self.sess.run(self.aloss, {self.tfs: s, self.tfa: a, self.tfadv: adv}))
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        print("closs",self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r}))

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1,100,tf.nn.relu,trainable=trainable)
            l3 = tf.layers.dense(l2,100,tf.nn.relu,trainable=trainable)
            a_prob = tf.layers.dense(l3, A_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, edit_s):
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: edit_s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        return action

    def get_v(self, edit_s):
        if edit_s.ndim < 2: edit_s = edit_s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: edit_s})[0, 0]

    def restore_para(self):
        var = tf.global_variables()
        var_to_restore = [val for val in var if 'pi' in val.name and 'Adam' not in val.name]
        saver = tf.train.Saver(var_to_restore)
        saver.restore(self.sess, "./model/model.ckpt")
