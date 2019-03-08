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
        self.train_num = 0
        self.replace_num = 300

        self.filename = "data_dqn.txt"
        self.f = open(self.filename)
        self.MAX_LINE = 6250789

        self.s_list = []
        self.a_list = []
        self.snext_list = []
        self.r_list = []
        self.done_list = []

        self.N_State = 292
        self.N_Action = 5
        self.tf_s = tf.placeholder(tf.float32, [None, self.N_State])
        self.tf_s_ = tf.placeholder(tf.float32, [None, self.N_State])
        self.tf_a = tf.placeholder(tf.int32, [None,])
        self.tf_r = tf.placeholder(tf.float32, [None,])

        self.q = self.build_Q_net("Qnet", trainable=True)
        self.q_next = self.build_Qnext_net("Qnext", trainable=False)
        self.input_q_values = tf.placeholder(tf.float32, [None], name='input_q_values')
        a_indices = tf.stack([tf.range(tf.shape(self.tf_a)[0], dtype=tf.int32), self.tf_a], axis=1)
        self.q_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)

        self.td_error = tf.reduce_mean(tf.squared_difference(self.input_q_values, self.q_wrt_a))
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.td_error)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def read_data(self):
        pointer = 0
        subdata = " "
        while(pointer < self.MAX_LINE):
            subdata += self.f.readline()
            pointer += 1

        singledata = str.split(subdata, '#')

        for ii in range(len(singledata)-1):
            state_i = []
            state_i_ = []
            tmp_end = singledata[ii].rfind("]", 0, 4000)
            tmp_state = singledata[ii][2:tmp_end]
            statespecific = tmp_state.split(" ")
            for item in statespecific:
                if item != '':
                    state_i.append(float(item))

            tmp_action = singledata[ii][tmp_end+2]

            tmp_start2 = singledata[ii].rfind("[", 100, -1)
            tmp_end2 = singledata[ii].rfind("]", 4000, -1)
            tmp_state_ = singledata[ii][tmp_start2+1:tmp_end2]

            statespecific2 = tmp_state_.split(" ")
            for item in statespecific2:
                if item != '':
                    state_i_.append(float(item))

            tmp_reward = singledata[ii][tmp_end2+2:-3]

            tmp_done = singledata[ii][-1]

            self.s_list.append(state_i)
            self.snext_list.append(state_i_)
            self.a_list.append(int(tmp_action))
            self.r_list.append(float(tmp_reward))
            self.done_list.append(int(tmp_done))

        print(len(self.s_list),len(self.snext_list),len(self.a_list),len(self.r_list),len(self.done_list))

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
        if self.train_num % self.replace_num == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qnext')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qnet')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])


        s_list, a_list, snext_list, r_list, done_list = self.read_batch()

        self.q_target = []
        self.q_eval = []
        a_indices = tf.stack([tf.range(tf.shape(self.tf_a)[0], dtype=tf.int32), self.tf_a], axis=1)
        self.q_eval = tf.gather_nd(params=self.q, indices=a_indices)
        all_q_next = self.q_next.eval(session=self.sess, feed_dict={self.tf_s_: snext_list})
        for i in range(self.BATCH_SIZE):
            if done_list[i] == 0:
                self.q_target.append(r_list[i] + self.GAMMA * np.max(all_q_next[i]))
            else:
                self.q_target.append(r_list[i])

        self.sess.run(self.train_op,
            {self.tf_s:s_list,self.tf_r:r_list, self.tf_s_:snext_list,
             self.tf_a:a_list,self.input_q_values:self.q_target})

        self.train_num += 1
        if self.train_num % 100 == 0:
            print(self.train_num,self.sess.run(self.td_error,
                {self.tf_s:s_list,self.tf_r:r_list, self.tf_s_:snext_list,
                 self.tf_a:a_list,self.input_q_values:self.q_target}))

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

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './dqn_model/model.ckpt')
        print("save done!")

myDQFD = DQFD()
myDQFD.read_data()
for i in range(100000):
    myDQFD.learn()
    if i == 99999:
        myDQFD.save()