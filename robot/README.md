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


焦虑已经过分了。
需要点别的东西来缓解一下的程度。
来列下一月计划，只要不被傻逼事情打乱那应该是做得到的。

接下来一个月的主题应该是：energy based model以及RL for trajectory optimization。至少如果这些顺利地话可以在最后几天赶下ICLR，虽然实在是没有必要的说。
要完成的baseline有graph neural network，以及众多model-based RL算法。

应该按照一定的节奏来才行，三天换个task（按照现在的经验，一个project一天显然是不够的。

那么接下来总共有四周的时间，总共有两天起步，八个周期，四个间歇日——写作业，顺便pre和teach——这样虽然不是效率最高的，但是至少可以活得开心点。

今明两天主要就是看看graph neural network的文章能够复现到什么程度。看看dm control到底能不能玩遛。这么来吧：

基本准则是sapien的task和我们的算法的不断迭代。时时刻刻push下Yiran，chutong和zhizuo。

- 12.30 - 1.1: Graph Neural Network，熟悉dm control，完成基础实验，包括对应的control baseline（这个现在超简单）。

- 1.2 - 1.4: PETS-CEM以及POLO，based on mbRL baseline。熟悉mujoco上的control task们。

- 1-6 - 1-8: 尝试sapien dataset。

- 1-9 - 1-11: 至少在这个时候能够在sapien上apply之前的model-based RL算法，同时调查multi-task RL的一些算法，以及baseline。

- 1-13 - 1-15: energy-based method for forward model on previous environment，至少可以实现baseline。这个时候chutong应该能干点活了吧。

- 1-16 - 1-18: RL based trajecotory optimization on previous model, with Chutong.

- 1-20 - 1-22: RL based trajectory optimization, together with energy-based model, with chutong. 构建带contact的hand manipulation task。

- 1-23 - 1-25: Contact Invariant Optimization, with neural network。
