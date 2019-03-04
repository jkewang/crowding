import tensorflow as tf
import numpy as np
import random
import math
import env

class PretrainActor(object):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.LR = 1e-3
        self.N_ACTIONS = 5
        self.N_S = 292

        self.tf_s = tf.placeholder(tf.float32, [None, self.N_S])
        self.tf_a = tf.placeholder(tf.float32, [None, 5])
        self.sess = tf.Session()

        self.learning_step = 0

        self.filename = "data.txt"
        self.f = open(self.filename)
        self.MAXLINE = 6250789
        self.s = []
        self.action = []

        self.a,self.a2 = self.build_actor()
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a, self.tf_a), axis=1))
        self.loss2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a2, self.tf_a),axis=1))
        self.train1 = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        self.train2 = tf.train.AdamOptimizer(self.LR).minimize(self.loss2)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def read_data(self):
        pointer = 0
        subdata = " "
        while (pointer < self.MAXLINE):
            subdata += self.f.readline()
            pointer += 1

        singledata = str.split(subdata, '#')
        for single in singledata:
            state_i = []
            a_i = []
            state =single[2:-3]
            statespecific = state.split(" ")
            for item in statespecific:
                if item != '':
                    state_i.append(float(item))

            if len(state_i) == 292:
                self.s.append(state_i)

            read_action = single[-1]
            try:
                Action = float(read_action)
                if Action == 0:
                    self.action.append([1.0, 0.0, 0.0, 0.0, 0.0])
                elif Action == 1:
                    self.action.append([0.0, 1.0, 0.0, 0.0, 0.0])
                elif Action == 2:
                    self.action.append([0.0, 0.0, 1.0, 0.0, 0.0])
                elif Action == 3:
                    self.action.append([0.0, 0.0, 0.0, 1.0, 0.0])
                else:
                    self.action.append([0.0, 0.0, 0.0, 0.0, 1.0])
            except:
                pass

        print(len(self.s),len(self.action))

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
        return a_prob,a_prob2

    def learn(self):
        self.learning_step += 1
        s,a = self.read_batch()
        self.sess.run(self.train1,{self.tf_s:s,self.tf_a:a})
        self.sess.run(self.train2,{self.tf_s:s,self.tf_a:a})
        if self.learning_step % 100 == 0:
            print("loss1 = ",self.sess.run(self.loss,{self.tf_s:s,self.tf_a:a}))
            print("loss2 = ",self.sess.run(self.loss2,{self.tf_s:s,self.tf_a:a}))

    def read_batch(self):
        s_list = []
        a_list = []
        for i in range(self.BATCH_SIZE):
            index = random.randint(0,len(self.s)-1)
            s_list.append(self.s[index])
            a_list.append(self.action[index])

        return s_list,a_list

    def choose_action(self, edit_s):
        prob_weights = self.sess.run(self.a, feed_dict={self.tf_s: edit_s[None, :]})
        print(prob_weights.shape[1])
        print(prob_weights.ravel())
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        print(action)
        return action

    def save(self):
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
        for v in self.ae_params:
            print(v.name)

        self.saver.save(self.sess,'./model/model.ckpt')
        print("save done!")

PA = PretrainActor()
PA.saver.restore(PA.sess,"./model/model.ckpt")
my_env = env.TrafficEnv()

for i_episode in range(1000000):
    # listener()
    s,rawOcc = my_env.reset()
    s = np.concatenate([s[0], np.reshape(s[1] + s[2], -1)])

    while True:
        #s[280] = np.random.rand()
        a = PA.choose_action(s)
        #a = myNaiveCon.gen_action(s,rawOcc)

        s_, r, done, _, rawOcc = my_env.step(a)
        s_ = np.concatenate([s_[0], np.reshape(s_[1] + s_[2], -1)])

        s = s_

        if done:
            break
