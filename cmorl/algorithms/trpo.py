import numpy as np

import torch
from torch.autograd import Variable
from cmorl.utils.utils import *
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

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x
def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):

    fval = f(True).data
    # print("fval---------------------------:", fval)
    # print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("fval after", newfval.item())
            return True, xnew
    return False, x

def linesearch_mo(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = fval
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew, stepfrac * fullstep
    return False, x, stepfrac * fullstep


def trpo_step(model, get_loss, get_kl, max_kl, damping):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)
    return loss


def trpo_step_mo(model, get_loss, get_kl, max_kl, damping):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    prev_params = get_flat_params_from(model)
    success, new_params, new_grads = linesearch_mo(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    return loss, new_grads

def cagrad(grads, grads2, c=0.5):
    g1 = (grads)
    g2 = (grads2)
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


class Cagrad_all():
    def __init__(self, c=0.5):
        self.cagrad_c = c
    def cagrad(self, grad_vec, num_tasks):
        """
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec

        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG)+1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True)
        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg+1e-4).sqrt() * self.cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward()
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)
        g = ((1/num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads).sum(0) / (1 + self.cagrad_c**2)
        return g

    def cagrad_exact(self, grad_vec, num_tasks):
        grads = grad_vec / 100.
        g0 = grads.mean(0)
        GG = grads.mm(grads.t())
        x_start = np.ones(num_tasks)/num_tasks
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.cpu().numpy()
        b = x_start.copy()
        c = (self.cagrad_c*g0.norm()).cpu().item()
        def objfn(x):
            return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + \
                    c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww= torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-4)
        g = (g0 + lmbda * gw) / (1 + lmbda)
        return g * 100

    def cagrad_fast(self, grad_vec, num_tasks):
        n = self.fast_n
        scale = 100.
        grads = grad_vec / scale
        GG = grads.mm(grads.t())
        g0_norm = (self.fast_w.view(1, -1).mm(GG).mm(self.fast_w.view(-1, 1))+1e-8).sqrt().item()

        x_start = np.ones(n) / n
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.cpu().numpy()
        c = self.cagrad_c*g0_norm
        def objfn(x):
            return (x.reshape(1,n).dot(A).dot(self.fast_w_numpy.reshape(n,1)) + \
                    c * np.sqrt(x.reshape(1,n).dot(A).dot(x.reshape(n,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww= torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = np.sqrt(w_cpu.reshape(1,n).dot(A).dot(w_cpu.reshape(n,1))+1e-8).item()
        lmbda = c / (gw_norm+1e-4)
        g = ((self.fast_w.view(-1,1)+ww.view(-1,1)*lmbda)*grads).sum(0)
        g = g / (1 + self.cagrad_c) * scale
        return g


class Cagrad_upgrade():
    def __init__(self, alpha=0.1, lam = 0.1):
        self.alpha = alpha
        self.lam = lam

    def cagrad_momentum(self, grads, grads2, c=0.5):
        g1 = (grads)
        g2 = (grads2)
        g0 = (g1 + g2) / 2

        g11 = g1.dot(g1).item()
        g12 = g1.dot(g2).item()
        g22 = g2.dot(g2).item()

        g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12 + 1e-4)
        coef = c * g0_norm

        def obj(x):
            return coef * np.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-4) + \
                   0.5 * x * (g11 + g22 - 2 * g12) + (0.5 + x) * (g12 - g22) + g22

        res = minimize_scalar(obj, bounds=(0, 1), method='bounded')

        x = res.x

        x = self.alpha * self.lam + (1 - self.alpha) * x
        self.lam = x

        gw = x * g1 + (1 - x) * g2
        gw_norm = np.sqrt(x ** 2 * g11 + (1 - x) ** 2 * g22 + 2 * x * (1 - x) * g12 + 1e-4)

        lmbda = coef / (gw_norm + 1e-4)
        g = g0 + lmbda * gw
        return g / (1 + c)



class CR_MOGM():
    def __init__(self, alpha=0.1, lam = 0.1):
        self.alpha = alpha
        self.lam = lam
        self.lambda_1 = 0
        self.lambda_2 = 0
        self.lam_prev_1 = 0
        self.lam_prev_2 = 0


    def cr_mogm_grad(self, grads, grads2, c=0.5):
        g1 = (grads)
        g2 = (grads2)
        g0 = (g1 + g2) / 2
        g11 = g1.dot(g1).item()
        g12 = g1.dot(g2).item()
        g22 = g2.dot(g2).item()
        self.lambda_hat_1 = 0.5 * (1 - (g1 * g2) / g11)  # lambda_1
        self.lambda_hat_2 = 0.5 * (1 - (g1 * g2) / g22)  # lambda_2
        if g12 < 0:
            u_pc_1 = (1 - (g1 * g2) / g22) * g1
            u_pc_2 = (1 - (g1 * g2) / g11) * g2
            self.lambda_hat_1 = self.lambda_hat_1 - u_pc_1
            self.lambda_hat_2 = self.lambda_hat_2 - u_pc_2
            self.lambda_new_1  = self.alpha * self.lam_prev_1 + (1 - self.alpha) * self.lambda_hat_1
            self.lambda_new_2 = self.alpha * self.lam_prev_2 + (1 - self.alpha) * self.lambda_hat_2

            self.lam_prev_1 = self.lambda_new_1
            self.lam_prev_2 = self.lambda_new_2

            d_k = (self.lambda_new_1 * g1 + self.lambda_new_2 * g2)
            return d_k
        else:
            # v04
            self.lambda_new_1 = self.alpha * self.lam_prev_1 + (1 - self.alpha) * self.lambda_hat_1
            self.lambda_new_2 = self.alpha * self.lam_prev_2 + (1 - self.alpha) * self.lambda_hat_2
            self.lam_prev_1 = self.lambda_new_1
            self.lam_prev_2 = self.lambda_new_2
            d_k = (self.lambda_new_1 * g1 + self.lambda_new_2 * g2)
            return d_k



