import abc
import numpy as np
from math import ceil


class Annealer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self):
        pass


class ConstantAnnealer(Annealer):
    def __init__(self, anneal_param):
        super(ConstantAnnealer, self).__init__()
        self.anneal_param = anneal_param

    def update(self):
        pass


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += step
            i += 1
    return L


class CyclicAnnealer(Annealer):
    """
    Inspired from https://arxiv.org/pdf/1903.10145.pdf
    """

    def __init__(self, anneal_param, M, R, T, policy_type='linear'):
        super(CyclicAnnealer, self).__init__()
        self.anneal_param = anneal_param
        self.base_val = anneal_param
        self.t = -1

        self.M = M
        self.R = R
        self.T = T
        if policy_type == 'linear':
            self.anneal_params = frange_cycle_linear(0.0, 1.0, n_epoch=T, n_cycle=M, ratio=R)
        elif policy_type == 'sigmoid':
            self.anneal_params = frange_cycle_sigmoid(0.0, 1.0, n_epoch=T, n_cycle=M, ratio=R)
        self.policy_type = policy_type

    def update(self):
        self.t += 1
        if self.policy_type in ['linear', 'sigmoid']:
            self.anneal_param = self.anneal_params[min([self.t, len(self.anneal_params)-1])]
        else:
            τ = self.tau()
            if τ <= self.R:
                self.anneal_param = min([τ + self.base_val, 1.0])
            else:
                self.anneal_param = 1.0

    def tau(self):
        return ((self.t - 1) % ceil(self.T / self.M)) / (self.T / self.M)
