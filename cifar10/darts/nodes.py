import os
import sys

import numpy as np
import torch
from torch import nn

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return inputs.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        grad_inputs = grad_output.clone()
        temp = abs(inputs - thresh) < lens
        return grad_inputs * temp.float()
        # grad = torch.exp((thresh - inputs)) / ((torch.exp((thresh - inputs)) + 1) ** 2)
        # return grad * grad_output


class LIFNode(nn.Module):
    def __init__(self,
                 shape=None,
                 device='cpu',
                 decay=1.):
        super(LIFNode, self).__init__()
        self.shape = shape
        self.device = device
        self.decay = decay
        self.act_fun = ActFun.apply

        self.mem = None
        self.spike = None
        self.reset()

    def reset(self):
        self.mem = None
        self.spike = None

    def forward(self, inputs):
        if self.mem is None:
            self.mem = torch.zeros_like(inputs, requires_grad=False)
        else:
            self.mem += inputs

        self.spike = self.mem.clone()
        self.spike[(self.spike < thresh) & (self.spike > -thresh)] = 0.
        self.mem = self.mem * self.decay
        self.mem[(self.mem >= thresh) | (self.mem <= -thresh)] = 0.
        return self.spike

    def get_fire_rate(self):
        return float((self.spike.detach() != 0.).sum()) / float(np.product(self.spike.shape))


class SoftLIFNode(nn.Module):
    def __init__(self,
                 shape,
                 device='cpu',
                 decay=1.):
        super(SoftLIFNode, self).__init__()
        self.shape = shape
        self.device = device
        self.decay = decay
        self.act_fun = nn.Sigmoid()

        self.mem = None
        self.spike = None
        self.reset()

    def reset(self):
        self.mem = torch.zeros(self.shape, device=self.device)
        self.spike = torch.zeros(self.shape, device=self.device)

    def forward(self, inputs):
        self.mem = self.mem * self.decay * (1. - self.spike) + inputs
        self.spike = self.act_fun(self.mem)
        return self.spike


class DirectNode(nn.Module):
    def __init__(self,
                 shape,
                 device='cpu',
                 decay=1.):
        super(DirectNode, self).__init__()
        self.shape = shape
        self.device = device

        self.spike = None
        self.reset()

    def reset(self):
        self.spike = torch.zeros(self.shape, device=self.device)

    def forward(self):
        pass

    def integral(self, inputs):
        self.spike += inputs
