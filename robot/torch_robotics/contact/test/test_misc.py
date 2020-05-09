import numpy as np
from robot.utils.plot_curves import plot_curves

def test_d():
    power = 2
    mid = 0.5
    N = 500
    x = (np.arange(N*2) - N)/(N*2) * 5
    y = 1/(1+np.exp(-(np.abs(x))**2))
    plot_curves({'sigmoid': (x, y)})



if __name__ == "__main__":
    test_d()