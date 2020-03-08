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

    grad_x_0 = torch.autograd.grad(y.sum(), x, only_inputs=True, retain_graph=True)[0]
    grad_x_1 = torch.autograd.grad((y**2).sum(), x, only_inputs=True)[0]
    grad_x = torch.stack((grad_x_0.detach(), grad_x_1.detach()))

    grad_x_2 = model.get_jacobian(x, torch.stack([y*0+1, 2*y]))
    print(((grad_x-grad_x_2)**2).sum())

if __name__ == '__main__':
    main()