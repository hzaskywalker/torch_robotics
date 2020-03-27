from robot import A

def main():
    args = A.train.get_args()
    args.lr = 0.001
    A.pets.pets_trainer(args)

if __name__ == '__main__':
    main()