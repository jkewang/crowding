import env
import naive_controller
import tensorflow as tf
import numpy as np
import random

class DAgger(object):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.LR = 1e-3
        self.N_ACTIONS = 5
        self.N_S = 292
        self.learning_step = 0
        self.memory_size = 3000
        self.memory_pointer = 0

        self.s = []
        self.action = []

        self.tf_s = tf.placeholder(tf.float32, [None, self.N_S])
        self.tf_a = tf.placeholder(tf.float32, [None, 5])
        self.sess = tf.Session()

        self.a, self.a2 = self.build_actor()
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a, self.tf_a), axis=1))
        self.loss2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a2, self.tf_a), axis=1))
        self.train1 = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        self.train2 = tf.train.AdamOptimizer(self.LR).minimize(self.loss2)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_actor(self):
        with tf.variable_scope("pi"):
            l1 = tf.layers.dense(self.tf_s, 100, tf.nn.relu)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu)
            l3 = tf.layers.dense(l2, 100, tf.nn.relu)
            a_prob = tf.layers.dense(l3, self.N_ACTIONS, tf.nn.softmax)
        with tf.variable_scope("oldpi"):
            l1 = tf.layers.dense(self.tf_s, 100, tf.nn.relu)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu)
            l3 = tf.layers.dense(l2, 100, tf.nn.relu)
            a_prob2 = tf.layers.dense(l3, self.N_ACTIONS, tf.nn.softmax)
        return a_prob, a_prob2

    def choose_action(self, edit_s):
        prob_weights = self.sess.run(self.a, feed_dict={self.tf_s: edit_s[None, :]})

        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())

        return action

    def store_transition(self, s, a):

        a_label = [0.0 for ii in range(5)]
        a_label[a] = 1.0

        if self.memory_pointer < 3000:
            self.action.append(a_label)
            self.s.append(s)
        else:
            self.action[self.memory_pointer % 3000] = a_label
            self.s[self.memory_pointer % 3000] = s
        self.memory_pointer += 1

    def read_batch(self):
        s_list = []
        a_list = []
        for i in range(self.BATCH_SIZE):
            index = random.randint(0, len(self.s) - 1)
            s_list.append(self.s[index])
            a_list.append(self.action[index])

        return s_list, a_list

    def learn(self):
        self.learning_step += 1
        s, a = self.read_batch()
        self.sess.run(self.train1, {self.tf_s: s, self.tf_a: a})
        self.sess.run(self.train2, {self.tf_s: s, self.tf_a: a})
        if self.learning_step % 100 == 0:
            print("loss1 = ", self.sess.run(self.loss, {self.tf_s: s, self.tf_a: a}))
            print("loss2 = ", self.sess.run(self.loss2, {self.tf_s: s, self.tf_a: a}))

    def save(self):
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
        for v in self.ae_params:
            print(v.name)

        self.saver.save(self.sess,'./model/model.ckpt')
        print("save done!")

my_env = env.TrafficEnv()
myNaiveCon = naive_controller.NaiveCon()
DA = DAgger()

epsilon = 0.1

for i_episode in range(10000):
    s, rawOcc = my_env.reset()
    s = np.concatenate([s[0], np.reshape(s[1] + s[2], -1)])
    ep_r = 0
    while True:
        #s[280] = np.random.rand()
        a = DA.choose_action(s)
        a_label = myNaiveCon.gen_action(s,rawOcc)

        rand_num = random.random()
        if epsilon < rand_num:
            s_, r, done, _, rawOcc = my_env.step(a_label)
        else:
            s_, r, done, _, rawOcc = my_env.step(a)

        ep_r += r
        DA.store_transition(s,a_label)

        if DA.memory_pointer > 3000 and DA.memory_pointer%30==0:
            DA.learn()

        s_ = np.concatenate([s_[0], np.reshape(s_[1] + s_[2], -1)])
        s = s_

        if epsilon <0.95:
            epsilon += 0.000001

        if done:
            print("now_episode:", i_episode, "ep_r:", ep_r, "now_epsilon:", epsilon)
            break

DA.save()