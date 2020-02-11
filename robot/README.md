# Robot Manipulation ToolBox

## Environment
- cartpole and pendulum, we have the totally differentiable environments.
- three/two link robot arm: we know the jacobian, and is modeled by graph. 
- dm control suite, several environments.
- mujoco robot arm, for motion planning.
- sapien robot arm, there is nothing now.
- [] The final goal should be a object manipulation environment.

It seems that we can get differentiable cost for all environments?

## Model
- The original model
- MLP
- GNN (only when the graph is given)
- [] Energy-based model
- [] Contact/search optimization
- [] Few-shot Model, which can output a posterior according to the previous trajectories.


## Controller
- GD controller, it seems very slow.
- iLQR controller.
- CEM
- Some motion planning algorithm


Environment -> Model -> Controller


