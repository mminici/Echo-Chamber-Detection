import torch
import torch.nn as nn


class EchoChamberLoss(nn.Module):
    def __init__(self, s, h, B, model_type='exact-posterior', anneal_param=1.0):
        super().__init__()

        self.s = s
        self.h = h
        self.B = B

        self.eps = 1e-10

        self.softmax_func = torch.nn.Softmax(dim=-1)
        self.relu_func = torch.nn.ReLU()
        self.anneal_param = anneal_param

        self.model_type = model_type

    def kld(self, q, p):
        return (q * (torch.log(q) - torch.log(p))).sum()

    def _kld(self, q, p):
        return (q * (torch.log(q) - torch.log(p))).sum(axis=1).mean()

    def forward(self, q_probs, p_probs, eta, y, is_link=True, p=None):
        loss_for_negatives = (q_probs * torch.log(1-p_probs)).sum(axis=-1)
        loss_for_positives = (q_probs * torch.log(p_probs)).sum(axis=-1)

        kld = torch.tensor(0.)

        if self.model_type != 'exact-posterior':
            if is_link:
                alpha = self.s * (1 - torch.abs(eta)) + self.h * torch.abs(eta)
                prior = alpha / alpha.sum(dim=-1)
                kld = self.kld(q_probs, prior)
            else:
                alpha = self.B * torch.abs(eta) + self.eps
                c_sim_to_prop = self.relu_func(torch.einsum('c, b->bc', eta, p))
                alpha = torch.einsum('c, bc -> bc', alpha, c_sim_to_prop)
                alpha += 1e-6
                prior = torch.einsum('bc, b -> bc', alpha, 1/alpha.sum(dim=-1))
                kld = self._kld(q_probs, prior)

        return -(y*loss_for_positives + (1-y)*loss_for_negatives), self.anneal_param*kld
