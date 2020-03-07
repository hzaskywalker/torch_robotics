import argparse
import torch
from robot.model.arm.lagrangian import grad_fc, grad_relu, grad_sequential, grad_softplus

def main():
    model = grad_sequential(
        grad_fc(3, 256),
        grad_relu(),
        grad_fc(256, 256),
        grad_relu(),
        grad_sequential(
            grad_softplus(),
            grad_fc(256, 256),
            grad_fc(256, 256),
            grad_relu(),
        ),
        grad_fc(256, 256),
        grad_softplus(),
        grad_fc(256, 8),
    )

    x = torch.rand((10, 3)).requires_grad_(True)

    y = model(x)

    grad_x = torch.autograd.grad(y.sum(), x, only_inputs=True)[0]

    grad_x_2 = model.get_jacobian(x, y*0+1)
    print(((grad_x.detach()-grad_x_2)**2).sum())

if __name__ == '__main__':
    main()