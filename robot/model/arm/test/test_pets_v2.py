from robot import A
import sys

def cheetah():
    args = A.train_utils.get_args()
    args.env_name = 'cheetah'
    args.path = 'pets_cheetah'
    args.lr = 0.001
    args.num_train_iter = 5
    A.pets.online_trainer(args)

def arm():
    args = A.train_utils.get_args()
    args.env_name = 'arm'
    args.path = 'pets_arm'
    args.lr = 0.001
    args.num_train_iter = 5
    A.pets.online_trainer(args)

if __name__ == '__main__':
    args = A.train_utils.get_args()
    if args.env_name == 'cheetah':
        cheetah()
    else:
        arm()