# the most simple model of the acrobat
# the input is transformed into sin/cos
import torch
from robot import A
import os
from .frame_types import SapienAcrobat2Frame

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

    def set_policy(self):
        self.set_rollout_model()
        args = self.args
        self.controller = A.train_utils.RolloutCEM(self.rollout_predictor, self.env.action_space, iter_num=5,
                                                   horizon=4, num_mutation=500, num_elite=20, device=args.device)

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
    else:
        max_velocity = 20 if args.env_name == 'diff_acrobat' else 100
        timestep = 0.1 if args.env_name == 'diff_acrobat' else 0.025
        model = ArmModel(2, dtype=torch.float, max_velocity=max_velocity, timestep=timestep)

    AcrobatTrainer(dataset, frame_type, model, args)
