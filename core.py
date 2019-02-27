import env
import time
import random
import dqn_fc as bt
import numpy as np
import logging

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

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
logger.addHandler(handler)

my_env = env.TrafficEnv()
#bt.saver.restore(bt.sess,"./model/my_light_model/my-model.ckpt-3500")

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
    #print(s_sliding,s_others)
    # fsm.Tick(command)
    k = 0
    ep_r = 0
    mydict = ["go","stop","left","right","nothing"]
    while True:
        actions_value = bt.choose_action(s_sliding, s_others)
        actions_value_mask = mask(actions_value)
        action = np.argmax(actions_value_mask)
        print(actions_value)
        print(actions_value_mask)
        print(mydict[action])

        # print("now_action",int(action))
        s, r, is_done, dist = my_env.step(action)
        s_pre_others2 = np.array(s[1] + s[2])
        for i in range(N_others):
            s_pre_others[i] = s_pre_others2[i % 12]

        s_sliding_, s_others_ = s[0], s_pre_others
        #print(s_others_)

        bt.store_transition(s_sliding, s_others, action, r, s_sliding_, s_others_, is_done)

        s_sliding = s_sliding_
        s_others = s_others_

        k += 1
        ep_r += r

        if (bt.EPSILON < 0.95):
            bt.EPSILON += 0.00003
        if (bt.MEMORY_COUNTER > bt.MEMORY_CAPACITY):
            bt.learn()
            if is_done:
                print("Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
        if is_done:
            logger.info("collecting! ------Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
            print("collecting! ------Ep:", i_episode, "|Ep_r:", round(ep_r, 2), "|Epsilon", bt.EPSILON)
            if i_episode % 200 == 0:
                bt.saver.save(bt.sess, './model/my-model.ckpt', global_step=i_episode)
            break