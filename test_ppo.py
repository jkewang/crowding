import env
import time
import random
import ppo
import numpy as np
import logging
import matplotlib.pyplot as plt
import naive_controller

my_env = env.TrafficEnv()
#bt.saver.restore(bt.sess,"./model/my_light_model/my-model.ckpt-3500")
all_ep_r = []
myppo = ppo.PPO()

myppo.saver.restore(myppo.sess,"./model/my-model.ckpt-4000")
myNaiveCon = naive_controller.NaiveCon()

for i_episode in range(1000000):
    # listener()
    s,rawOcc = my_env.reset()
    s = np.concatenate([s[0], np.reshape(s[1] + s[2], -1)])

    buffer_s, buffer_a, buffer_r = [], [], []

    k = 0
    ep_r = 0
    mydict = ["go","stop","left","right","nothing"]
    while True:
        a = myppo.choose_action(s)
        #a = myNaiveCon.gen_action(s,rawOcc)

        s_, r, done, _, rawOcc = my_env.step(a)
        s_ = np.concatenate([s_[0], np.reshape(s_[1] + s_[2], -1)])

        s = s_
        ep_r += r

        if done:
            break

    if i_episode == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
    print('Ep: %i' % i_episode,"|Ep_r: %i" % ep_r,)