import env
import time
import random
import ppo
import numpy as np
import logging
import matplotlib.pyplot as plt

def mask(actions_value):
    print(s_others[3],s_others[4])
    q_mask = [0, 0, 0, 0, 0]
    if s_others[3] == 0:
        q_mask[3] -= 1000
    if s_others[4] == 0:
        q_mask[2] -= 1000

    actions_value_mask = [0,0,0,0,0]
    for i in range(len(actions_value)):
        actions_value_mask[i] = actions_value[i] + q_mask[i]

    return actions_value_mask

my_env = env.TrafficEnv()
#bt.saver.restore(bt.sess,"./model/my_light_model/my-model.ckpt-3500")
all_ep_r = []
myppo = ppo.PPO()

for i_episode in range(1000000):
    # listener()
    s = my_env.reset()
    N_others = 12*10
    s_pre_others = np.zeros((N_others))
    s_pre_others2 = np.array(s[1]+s[2])
    #print(s_pre_others2)
    for i in range(N_others):
        s_pre_others[i] = s_pre_others2[i%12]

    s_sliding, s_others = s[0],s_pre_others
    buffer_s, buffer_a, buffer_r = [], [], []

    k = 0
    ep_r = 0
    mydict = ["go","stop","left","right","nothing"]
    while True:
        a = myppo.choose_action(s)
        s_, r, done, _ = my_env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)
        s = s_
        ep_r += r

        if (k + 1) % ppo.BATCH == 0 or done:
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
        if done:
            break

    if i_episode == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
    print('Ep: %i' % i_episode,"|Ep_r: %i" % ep_r,)
    if i_episode % 1000 == 0:
        print("in!!!")
        xx = range(len(all_ep_r))
        plt.plot(xx,all_ep_r,'*')
        plt.show()
        plt.pause(10)
    break