from torch.autograd import Function
import torch
import torch.nn as nn
from sympy import *
import math
import sympy
import numpy as np


def weight_avg(prob):
    avg = 0
    for i in range(prob.size()[1]):
        avg += (i + 1) * prob[0][i]

    return avg


def find_para(avg):
    x = symbols('x')
    ans = solve(((1-avg.item())*sympy.exp(x)+(2-avg.item())*sympy.exp(2*x)+(3-avg.item())*sympy.exp(3*x)+(4-avg.item())\
                 *sympy.exp(4*x)+(5-avg.item())*sympy.exp(5*x)), x)

    return ans[0]


def v_prob(para):
    para = np.array(para, dtype=np.float64)
    denominator = 0
    v = []
    for i in range(5):
        denominator += np.exp((i+1)*para)

    for j in range(5):
        numerator = np.exp((j+1)*para)
        v.append(numerator/denominator)

    return torch.Tensor(v)


def u_prob(prob):
    u = list()
    len = prob.size()[1]
    for _ in range(len):
        u.append(1/len)

    return torch.Tensor(u)


def single_distance(p, q, r=2):

    assert p.shape == q.shape, "Length of the two distribution must be the same"
    cdf_p = torch.cumsum(p, dim=0)
    cdf_q = torch.cumsum(q, dim=0)
    cdf_diff = torch.abs(cdf_p - cdf_q)
    cdf_diff = torch.mean(cdf_diff ** r)

    return cdf_diff ** (1. / r)


def distance(p, q, r=2):

    # print(p.shape)
    # print(q.shape)
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_distance(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer
