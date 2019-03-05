import env
import time
import random
import ppo
import numpy as np
import logging
import matplotlib.pyplot as plt
import naive_controller

my_env = env.TrafficEnv()
all_ep_r = []
all_aloss = []
all_closs = []

myppo = ppo.PPO()
myNaiveCon = naive_controller.NaiveCon()

myppo.restore_para()
epsilon = 0.1
train_num = 0
for i_episode in range(1000000):
    # listener()
    s,rawOcc = my_env.reset()
    s = np.concatenate([s[0], np.reshape(s[1] + s[2], -1)])

    buffer_s, buffer_a, buffer_r = [], [], []

    k = 0
    ep_r = 0
    mydict = ["go","stop","left","right","nothing"]
    while True:

        ranA = np.random.rand()

        if ranA < 1:
            a = myppo.choose_action(s)
        else:
            a = myNaiveCon.gen_action(s,rawOcc)

        s_, r, done, _,rawOcc = my_env.step(a)
        s_ = np.concatenate([s_[0], np.reshape(s_[1] + s_[2], -1)])

        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)
        s = s_
        ep_r += r

        if (k + 1) % ppo.BATCH == 0 or done:
            if done:
                v_s_ = 0
            else:
                v_s_ = myppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + ppo.GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            myppo.update(bs, ba, br)
        k += 1
        if epsilon < 0.99:
            epsilon += 0.00003
        if done:
            if i_episode % 200 == 0:
                myppo.saver.save(myppo.sess, '/home/jkwang/PycharmProjects/crowding/model/my-model2.ckpt', global_step=i_episode)
            if i_episode % 200 == 0:
                print("in!!!")
                xx = range(len(all_ep_r))
                plt.plot(xx, all_ep_r, '*')
                plt.show()
                plt.pause(10)
            break

    if i_episode == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)

    print('Ep: %i' % i_episode,"|Ep_r: %i" % ep_r,"|epsilon:",epsilon)
