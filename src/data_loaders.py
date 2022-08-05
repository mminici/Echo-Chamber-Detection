import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, u, v, p, y, device=None):
        self.u = u
        self.v = v
        self.p = p
        self.y = y
        if device is not None:
            self.u = self.u.to(device)
            self.v = self.v.to(device)
            self.p = self.p.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.u.shape[0]

    def __getitem__(self, idx):
        """Generates one sample of data"""
        return self.u[idx], self.v[idx], self.p[idx], self.y[idx]


class PairwiseDataset(torch.utils.data.Dataset):

    def __init__(self, u, v, p, device=None):
        self.u_pos, self.u_neg = u
        self.v_pos, self.v_neg = v
        self.p_pos, self.p_neg = p
        if device is not None:
            self.u_pos = self.u_pos.to(device)
            self.v_pos = self.v_pos.to(device)
            self.p_pos = self.p_pos.to(device)
            self.u_neg = self.u_neg.to(device)
            self.v_neg = self.v_neg.to(device)
            self.p_neg = self.p_neg.to(device)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.u_pos.shape[0]

    def __getitem__(self, idx):
        """Generates one sample of data"""
        return self.u_pos[idx], self.v_pos[idx], self.p_pos[idx], self.u_neg[idx], self.v_neg[idx], self.p_neg[idx]
