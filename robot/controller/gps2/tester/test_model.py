import numpy as np
from robot.model.gmm_linear_model import GaussianLinearModel

def main():
    model = GaussianLinearModel(regularization=1e-6, prior_strength=10., min_samples_per_cluster=40,
                                max_clusters=50, max_samples=3000)

    dim = 4
    X = np.random.normal(size=(100, 4))
    y = np.stack((X.mean(axis=1), X.sum(axis=1)), axis=1) + np.random.normal(size=(100, 2)) * 0.3
    #y[:50] += 1
    y+=1

    model.update(X, y)
    x1 = np.random.normal(size=(2, 4))
    y1 = np.stack((x1.mean(axis=1), x1.sum(axis=1)), axis=1) + 1 + np.random.normal(size=(x1.shape[0], 2)) * 0.3
    model = model.eval(x1, y1)
    print(model.sample(x1[0]), y1[0])
    print(model)

if __name__ == '__main__':
    main()
