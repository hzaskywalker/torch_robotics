import cv2
import numpy as np
from robot.envs.hyrule import load_json, load_scene, make


env = make('x.json')
img = env.render(mode='rgb_array')


for i in range(1000):
    params = load_json(f'scenes/{i}.json')
    print(params['box']['center'])

    load_scene(env, params)
    img2 = env.render(mode='rgb_array')

    cv2.imshow('x', np.concatenate((img, img2), axis=1) )

    cv2.waitKey(0)
