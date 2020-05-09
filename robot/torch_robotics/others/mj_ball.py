import mujoco_py
import numpy as np
import cv2

model = mujoco_py.load_model_from_path('mj_ball.xml')
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjRenderContextOffscreen(sim, -1)

width = 500
height =500
print(sim.data.qpos.flat[:])

def get():
    return sim.data.qpos.flat[:], sim.data.qvel.flat[:]

def set(a, b):
    old_state = sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time, a, b,
                                     old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()

for i in range(20):
    set(np.array([0.1+i/100. * 0.001]), np.array([-0.1]))
    s = get()[0]

    for i in range(200):
        sim.step()
        if False:
            img = viewer.render(width, height, camera_id=None)
            img = viewer.read_pixels(width, height, depth=False)
            cv2.imshow('img', img[::-1])
            cv2.waitKey(1)
        print('xx', get())
    exit(0)
    print(get())
    #exit(0)
