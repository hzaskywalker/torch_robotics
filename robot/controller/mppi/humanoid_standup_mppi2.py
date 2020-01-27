#from trajopt.algos.mppi import MPPI
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
import cv2
from robot.controller.mppi.mppi2 import MPPIController
from robot.controller.mppi.utils import get_environment

# =======================================
ENV_NAME = 'humanoid_standup'
PICKLE_FILE = ENV_NAME + '_mppi.pickle'
SEED = 12345 - 2
N_ITER = 1
#H_total = 2 #100
H_total = 100
# =======================================

e = get_environment(ENV_NAME)
e.reset_model(seed=SEED)
mean = np.zeros(e.action_dim)
sigma = 1.0*np.ones(e.action_dim)
filter_coefs = [sigma, 0.25, 0.8, 0.0]

#agent = MPPI(e, H=30, paths_per_cpu=40, num_cpu=1,
#             kappa=25.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
#            default_act='mean', seed=SEED)
ts = timer.time()

def get_state(e):
    out = e.get_env_state()
    out = np.concatenate((out['qp'], out['qv']))
    return out
def set_state(e, s):
    d = len(e.init_qpos)
    e.sim.reset()
    e.set_state(s[:d], s[d:])
    e.sim.forward()

start = get_state(e)
print(start)
N_ITER = 5
agent = MPPIController(16, e.action_space, get_environment, ENV_NAME,
                       1, N_ITER, 200, kappa=25., gamma=1., sigma=1., num_process=20,
                       beta_0=0.25, beta_1=0.8, beta_2=0.0, seed=SEED)


actions2 = pickle.load(open(PICKLE_FILE, 'rb')).sol_act
if True:
    actions = []
    reward = 0
    for t in tqdm(range(H_total)):
        s = get_state(e)
        a = agent(s)
        _, r, _, _ = e.step(a)
        actions.append(a)
        reward += r
        #print(actions2[t] - a)
        #exit(0)
        #if t == 6:
        #    break
        if t % 25 == 0 and t > 0:
            print("==============>>>>>>>>>>> saving progress ")
            pickle.dump(actions, open(PICKLE_FILE+'2', 'wb'))

    pickle.dump(actions, open(PICKLE_FILE+'2', 'wb'))
    print("Trajectory reward = %f" % reward)

#actions = pickle.load(open(PICKLE_FILE, 'rb')).sol_act

print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))

# wait for user prompt before visualizing optimized trajectories
_ = input("Press enter to display optimized trajectory (will be played 10 times) : ")

import cv2
r = 0
for _ in range(1):
    set_state(e, start)
    for a in actions:
        img = e.render(mode='rgb_array')
        cv2.imshow('x', img)
        cv2.waitKey(1)
        r += e.step(a)[1]

    img = e.render(mode='rgb_array')
    cv2.imshow('x', img)
print('reward', r)
