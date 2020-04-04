from robot import A

def main():
    args = A.train.get_args()
    args.env_name = 'cheetah'
    args.lr = 0.001
    args.num_train_iter = 5
    A.pets.online_trainer(args)

if __name__ == '__main__':
    main()