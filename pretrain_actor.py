import tensorflow as tf
import numpy as np
import random
import math

class PretrainActor(object):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.LR = 1e-5
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

        self.a, self.a2 = self.build_actor()
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a, self.tf_a), axis=1))
        self.loss2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.a2, self.tf_a), axis=1))
        self.train1 = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        self.train2 = tf.train.AdamOptimizer(self.LR).minimize(self.loss2)
        self.sess.run(tf.global_variables_initializer())

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
                #state_i[280] = np.random.rand()
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

        add_action = []
        add_state = []
        for index in range(len(self.action)):
            if self.action[index][2] == 1 or self.action[index][3] == 1:
                for i in range(100):
                    add_action.append(self.action[index])
                    add_state.append(self.s[index])

        self.action.extend(add_action)
        self.s.extend(add_state)

        print(len(self.s), len(self.action))

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
        s,a = self.read_batch()
        self.sess.run(self.train1,{self.tf_s:s,self.tf_a:a})
        self.sess.run(self.train2,{self.tf_s:s,self.tf_a:a})
        if self.learning_step % 100 == 0:
            print("loss1 = ",self.sess.run(self.loss,{self.tf_s:s,self.tf_a:a}))
            print("loss2 = ",self.sess.run(self.loss2,{self.tf_s:s,self.tf_a:a}))
            #print(self.sess.run(self.a,{self.tf_s:s}))
            #print(a)
        self.learning_step += 1

    def read_batch(self):
        s_list = []
        a_list = []
        for i in range(self.BATCH_SIZE):
            index = random.randint(0,len(self.s)-1)
            s_list.append(self.s[index])
            a_list.append(self.action[index])

        return s_list,a_list

    def save(self):
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
        for v in self.ae_params:
            print(v.name)

        saver = tf.train.Saver()
        saver.save(self.sess,'./model/model.ckpt')
        print("save done!")

PA = PretrainActor()
PA.read_data()
for i in range(1000000):
    PA.learn()
    if i == 999999:
        PA.save()