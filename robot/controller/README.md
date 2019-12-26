# Control Algorithm

There are a lot of control algorithms and we will include them here.

- MPC (iLQR): we use the implementation here: https://locuslab.github.io/mpc.pytorch/ 
- NaiveGradientDescent
- CEM
- TD3: TD3 is somewhat stupid as we have to build the whole environment. 

We include some interfaces in forward_controller for the control algorithm with a forward algorithm.