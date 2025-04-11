from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
from tqdm import tqdm
from scipy.optimize import minimize, Bounds, minimize_scalar

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import seaborn as sns
import sys


LOWER = 0.000005

class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([
            [-3.0, 0],
            [3.0, 0]])

    def forward(self, x, compute_grad=False):
        x1 = x[0]
        x2 = x[1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2

        f = torch.tensor([f1, f2])
        if compute_grad:
            # print("test---------2---------") # used
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            return f, g
        else:
            return f

    def batch_forward(self, x):
        # print("test-------------------1------------------------") # not used
        x1 = x[:,0]
        x2 = x[:,1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2

        f  = torch.cat([f1.view(-1, 1), f2.view(-1,1)], -1)
        return f

def cagrad(grads, c=0.5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g0 = (g1+g2)/2

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)
    coef = c * g0_norm

    def obj(x):
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    return g / (1+c)


### Define the problem ###
F = Toy()

maps = {
    "sgd": mean_grad,
    "cagrad": cagrad,
    "mgd": mgd,
    "pcgrad": pcgrad,
}

def run_all():
    all_traj = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([-8.5, -5.]),
        torch.Tensor([9.,   9.]),
    ]

    for i, init in enumerate(inits):
        for m in tqdm(["sgd", "mgd", "pcgrad", "cagrad"]):
            all_traj[m] = None
            traj = []
            solver = maps[m]
            x = init.clone()
            x.requires_grad = True

            n_iter = 100000
            opt = torch.optim.Adam([x], lr=0.001)

            for it in range(n_iter):
                traj.append(x.detach().numpy().copy())

                f, grads = F(x, True)
                if m == "cagrad":
                    g = solver(grads, c=0.5)
                else:
                    g = solver(grads)
                opt.zero_grad()
                x.grad = g
                opt.step()

            all_traj[m] = torch.tensor(traj)
        torch.save(all_traj, f"toy{i}.pt")


def plot_results():
    plot3d(F)
    plot_contour(F, 1, name="toy_task_1")
    plot_contour(F, 2, name="toy_task_2")
    t1 = torch.load(f"toy0.pt")
    t2 = torch.load(f"toy1.pt")
    t3 = torch.load(f"toy2.pt")

    length = t1["sgd"].shape[0]

    for method in ["sgd", "mgd", "pcgrad", "cagrad"]:
        ranges = list(range(10, length, 1000))
        ranges.append(length-1)
        for t in tqdm(ranges):
            plot_contour(F,
                         task=0, # task == 0 meeas plot for both tasks
                         traj=[t1[method][:t],t2[method][:t],t3[method][:t]],
                         plotbar=(method == "cagrad"),
                         name=f"./imgs/toy_{method}_{t}")


if __name__ == "__main__":
    run_all()




