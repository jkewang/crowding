import env
import time
import random
import ppo
import numpy as np
import logging
import matplotlib.pyplot as plt
import naive_controller

my_env = env.TrafficEnv()
myNaiveCon = naive_controller.NaiveCon()
f1 = open('data_dqn.txt','w')

for i_episode in range(1000):
    # listener()
    s,rawOcc = my_env.reset()
    s = np.concatenate([s[0], np.reshape(s[1] + s[2], -1)])

    buffer_s, buffer_a, buffer_r = [], [], []

    k = 0
    ep_r = 0
    mydict = ["go","stop","left","right","nothing"]
    while True:
        a = myNaiveCon.gen_action(s,rawOcc)
        s_, r, done, _,rawOcc = my_env.step(a)
        s_ = np.concatenate([s_[0], np.reshape(s_[1] + s_[2], -1)])

        k += 1
        f1.write(str(s))
        f1.write("\n")
        f1.write(str(a))
        f1.write("\n")
        f1.write(str(s_))
        f1.write("\n")
        f1.write(str(r))
        f1.write("\n")
        f1.write(str(done))
        f1.write("#\n")

        s = s_
        if done:
            break

f1.close()