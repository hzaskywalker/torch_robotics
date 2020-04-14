# the most simple model of the acrobat
# the input is transformed into sin/cos
import torch
from robot import A
import os

class Agent(A.train_utils.RolloutAgent):
    # we now ask
    def get_loss(self, state, actions, future):
        predict, _ = self.rollout(state, actions, None)
        losses = predict.calc_loss(future)
        if 'gt_ee_loss' in self.loss_weights:
            predict.ee = self.model.fk(future.q.reshape(-1, future.q.shape[-1]), dim=2
                                       ).reshape(predict.ee.shape)
            losses['gt_ee_loss']  = predict.calc_loss(future)['ee_loss']
        return predict, losses


class AcrobatTrainer(A.trainer):
    def __init__(self, dataset, frame_type, model, args):
        if args.env_name != 'acrobat2':
            args.env_name = 'diff_acrobat'
        self.dataset = dataset
        self.frame_type = frame_type
        self.model = model
        self._dataset = dataset
        super(AcrobatTrainer, self).__init__(args)

    def set_model(self):
        return self.model

    def set_env(self):
        args = self.args
        self.env = A.train_utils.make(args.env_name)
        self.dataset = A.train_utils.DatasetWrapper(self._dataset, batch_size=args.batch_size,
                                                    timestep=args.timestep, cls = self.frame_type)

    def set_agent(self):
        args = self.args
        self.agent = Agent(self.model, lr=args.lr, loss_weights={
            'q_loss': args.weight_q,
            'dq_loss': args.weight_dq,
            'ee_loss': args.weight_ee * 0,
            'gt_ee_loss': args.weight_ee,
        }).cuda()

    def set_policy(self):
        self.set_rollout_model()
        self.controller = A.train_utils.RolloutCEM(self.rollout_predictor,
                                                   self.env.action_space, iter_num=5,
                                                   horizon=4, num_mutation=500,
                                                   num_elite=20, device=self.args.device)

DatasetConfigs = {
}

def assign_args(args, kwargs):
    for key, val in kwargs.items():
        assert key in vars(args), f"{key} not in args"
        args.__dict__[key] = val

def evaluate_model(env_name, model, **kwargs):
    from .mlp_model import MLP_ACROBAT
    from .phys_model import ArmModel
    from .frame_types import FRAMETYPES

    # ------------------------- handle args -----------------------------
    args = A.train_utils.get_args()
    args.num_train_iter = 2000
    args.num_valid_iter = 20
    args.timestep = 2
    args.lr = 0.01
    args.env_name = env_name

    if env_name in DatasetConfigs:
        assign_args(args, DatasetConfigs[env_name])
    assign_args(args, kwargs)

    # ------------------------ dataset -----------------------------------
    assert env_name in ['diff_acrobat2', 'acrobat2']
    dataset_path = os.path.join('/dataset/', env_name)
    dataset = A.train_utils.Dataset(dataset_path, device=args.device)

    # ------------------------- frame type -------------------------------
    frame_type = FRAMETYPES[env_name]

    # ------------------------- model  -----------------------------------
    if model == 'mlp':
        model = MLP_ACROBAT(frame_type.input_dims, frame_type.output_dims,
                                 4, 256, batchnorm=args.batchnorm)
    elif model == 'phys_gt':
        from robot.model.arm.exp.qacc import build_diff_model
        #model = build_diff_model()
        model = torch.load('model_gt.pkl')
    elif model == 'phys':
        max_velocity = 20 if args.env_name == 'diff_acrobat2' else 100
        timestep = 0.1 if args.env_name == 'diff_acrobat2' else 0.025
        damping = 0. if args.env_name == 'diff_acrobat2' else 0.5
        model = ArmModel(2, dtype=torch.float, max_velocity=max_velocity, timestep=timestep, damping=damping)
    else:
        raise NotImplementedError

    AcrobatTrainer(dataset, frame_type, model, args)
