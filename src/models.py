import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv

EC_PRIOR_SIZE = 16
SOCIAL_PRIOR_SIZE = 16
PROPAGATION_PRIOR_SIZE = 16
DEFAULT_ABLATION = "prop+link"
LINK_ABLATION = "prop"
PROP_ABLATION = "link"
DEFAULT_TRAINING_TYPE = 'normal'
PAIRWISE_TRAINING_TYPE = 'pairwise'


class MyTanh(nn.Module):
    def __init__(self, coef):
        super(MyTanh, self).__init__()
        self.coef = coef
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.coef * x)


class MyClamp(nn.Module):
    def __init__(self, tensor):
        super(MyClamp, self).__init__()
        self.tensor = nn.Parameter(tensor)

    def forward(self):
        return torch.clamp(self.tensor, -1., 1.)


class MyTanhTensor(nn.Module):
    def __init__(self, tensor, coef=1.0):
        super(MyTanhTensor, self).__init__()
        self.tensor = nn.Parameter(tensor)
        self.tanh = nn.Tanh()
        self.coef = coef

    def forward(self):
        return self.tanh(self.coef * self.tensor)


class MyEmbedding(nn.Module):
    def __init__(self, num_users, num_communities, device):
        super(MyEmbedding, self).__init__()
        self.embedding = nn.Sequential(*[
            nn.Embedding(num_users, num_communities),
            # nn.Softmax(dim=1)
        ])
        # self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        # self.full_index = torch.tensor(range(num_users), device=device, requires_grad=False)
        # nn.init.uniform_(self.embedding.weight, 1e-8, 1.0)

    def forward(self, x):
        return self.sigmoid(self.embedding(x))


class MySigTensor(nn.Module):
    def __init__(self, tensor):
        super(MySigTensor, self).__init__()
        self.tensor = nn.Parameter(tensor)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.tensor)[x]


class ECD(nn.Module):
    def __init__(self, num_users, num_communities, neighbors, device=None):
        super().__init__()
        self.link_neighbors, self.prop_neighbors = neighbors

        # self.theta = nn.Sequential(*[
        #     nn.Embedding(num_users, num_communities),
        # nn.Softmax(dim=1)
        # ])
        self.theta = MyEmbedding(num_users, num_communities, device)

        # self.phi = nn.Sequential(*[
        #    nn.Embedding(num_users, num_communities),
        # nn.Softmax(dim=1)
        # ])
        self.phi = MyEmbedding(num_users, num_communities, device)

        r1, r2 = [-1, 1]
        eta_tensor = (r1 - r2) * torch.rand(num_communities) + r2
        self.eta = MyTanhTensor(eta_tensor, coef=1.0)  # MyClamp(eta_tensor)

        # FIXME: questa potrebbe essere una GNN che sfrutta il grafo dei neighbors
        self.q_net_l = nn.Sequential(*[
            nn.Linear(2 * num_users, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, num_communities),
            nn.Softmax(dim=1)
        ])

        self.q_net_f = nn.Sequential(*[
            nn.Linear(2 * num_users, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, num_communities),
            nn.Softmax(dim=1)
        ])

        self.softmax_func = nn.Softmax(dim=1)
        self.relu_func = nn.ReLU()
        self.eps = 1e-10

    def forward(self, u, v, p=None):
        if self.p is None:
            q_probs, p_probs, eta = self.forward_link(u, v)
        else:
            q_probs, p_probs, eta = self.forward_prop(u, v, p)

        return q_probs, p_probs, eta

    def forward_link(self, u, v, use_softmax=False):
        x_u = self.link_neighbors[u]
        x_v = self.link_neighbors[v]

        x = torch.cat((x_u, x_v), -1)

        q_probs = self.q_net_l(x) + self.eps

        p_probs = (torch.abs(self.eta())) * self.theta(u) * self.theta(v) + \
                  (1 - torch.abs(self.eta())) * self.phi(u) * self.phi(v)

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        return q_probs, p_probs, self.eta()

    def forward_propagation(self, u, v, p, use_softmax=False):
        x_u = self.prop_neighbors[u]
        x_v = self.prop_neighbors[v]

        x = torch.cat((x_u, x_v), -1)

        q_probs = self.q_net_f(x) + self.eps

        p_probs = self.relu_func(torch.einsum('k, i->ik', self.eta(), p)) * self.theta(u) * self.theta(v)

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        return q_probs, p_probs, self.eta()


class AltMyEmbedding(nn.Module):
    def __init__(self, num_users, num_communities, device):
        super(AltMyEmbedding, self).__init__()
        self.embedding = nn.Sequential(*[
            nn.Embedding(num_users, num_communities)
        ])
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        self.full_index = torch.tensor(range(num_users), device=device, requires_grad=False)
        # nn.init.uniform_(self.embedding.weight, 1e-8, 1.0)

    def forward(self, x):
        return self.softmax(self.embedding(self.full_index))[x]


class AltECD(nn.Module):
    def __init__(self, num_users, num_communities, neighbors, device=None):
        super().__init__()
        self.link_neighbors, self.prop_neighbors = neighbors

        # self.theta = nn.Sequential(*[
        #     nn.Embedding(num_users, num_communities),
        #     nn.Softmax(dim=1)
        # ])
        self.theta = AltMyEmbedding(num_users, num_communities, device)

        # self.phi = nn.Sequential(*[
        #    nn.Embedding(num_users, num_communities),
        #    nn.Softmax(dim=1)
        # ])
        self.phi = AltMyEmbedding(num_users, num_communities, device)

        r1, r2 = [-1, 1]
        eta_tensor = (r1 - r2) * torch.rand(num_communities) + r2
        self.eta = MyTanhTensor(eta_tensor, coef=1.0)  # MyClamp(eta_tensor)

        self.q_net_l = nn.Sequential(*[
            nn.Linear(2 * num_users, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, num_communities),
            nn.Softmax(dim=1)
        ])

        self.q_net_f = nn.Sequential(*[
            nn.Linear(2 * num_users, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, num_communities),
            nn.Softmax(dim=1)
        ])

        self.softmax_func = nn.Softmax(dim=1)
        self.relu_func = nn.ReLU()
        self.eps = 1e-10

    def forward(self, u, v, p=None):
        if self.p is None:
            q_probs, p_probs, eta = self.forward_link(u, v)
        else:
            q_probs, p_probs, eta = self.forward_prop(u, v, p)

        return q_probs, p_probs, eta

    def forward_link(self, u, v, use_softmax=False):
        x_u = self.link_neighbors[u]
        x_v = self.link_neighbors[v]

        x = torch.cat((x_u, x_v), -1)

        q_probs = self.q_net_l(x) + self.eps

        p_probs = (torch.abs(self.eta())) * self.theta(u) * self.theta(v) + \
                  (1 - torch.abs(self.eta())) * self.phi(u) * self.phi(v)

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        return q_probs, p_probs, self.eta()

    def forward_propagation(self, u, v, p, use_softmax=False):
        x_u = self.prop_neighbors[u]
        x_v = self.prop_neighbors[v]

        x = torch.cat((x_u, x_v), -1)

        q_probs = self.q_net_f(x) + self.eps

        p_probs = self.relu_func(torch.einsum('k, i->ik', self.eta(), p)) * self.theta(u) * self.theta(v)

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        return q_probs, p_probs, self.eta()


class GCN(torch.nn.Module):
    def __init__(self, node_features, num_classes, dropout=.3):
        super().__init__()
        self.conv1 = GCNConv(node_features, 1024)
        self.conv2 = GCNConv(1024, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return self.softmax(x)


class PhiGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes, dropout=.3):
        super().__init__()
        self.conv1 = GCNConv(node_features, 1024)
        self.conv2 = GCNConv(1024, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return self.sigmoid(x)


class GCN_ECD(nn.Module):
    def __init__(self, num_users, num_communities, neighbors, edge_index, node_features, device=None):
        super().__init__()
        self.link_neighbors, self.prop_neighbors = neighbors
        self.edge_index = edge_index
        self.node_features = node_features

        self.theta = GCN(num_users, num_communities)

        self.phi = PhiGCN(num_users, num_communities)

        if device is not None:
            self.theta = self.theta.to(device)
            self.phi = self.phi.to(device)

        r1, r2 = [-1, 1]
        eta_tensor = (r1 - r2) * torch.rand(num_communities) + r2
        self.eta = MyTanhTensor(eta_tensor, coef=1.0)

        self.q_net_l = nn.Sequential(*[
            nn.Linear(2 * num_users, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, num_communities),
            nn.Softmax(dim=1)
        ])

        self.q_net_f = nn.Sequential(*[
            nn.Linear(2 * num_users, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, num_communities),
            nn.Softmax(dim=1)
        ])

        self.softmax_func = nn.Softmax(dim=1)
        self.relu_func = nn.ReLU()
        self.eps = 1e-10

    def forward(self, u, v, p=None):
        if self.p is None:
            q_probs, p_probs, eta = self.forward_link(u, v)
        else:
            q_probs, p_probs, eta = self.forward_prop(u, v, p)

        return q_probs, p_probs, eta

    def forward_link(self, u, v, use_softmax=False):
        x_u = self.link_neighbors[u]
        x_v = self.link_neighbors[v]

        x = torch.cat((x_u, x_v), -1)

        q_probs = self.q_net_l(x) + self.eps

        θ = self.theta(self.node_features, self.edge_index)
        φ = self.phi(self.node_features, self.edge_index)
        p_probs = (torch.abs(self.eta())) * θ[u] * θ[v] + (1 - torch.abs(self.eta())) * φ[u] * φ[v]

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        return q_probs, p_probs, self.eta()

    def forward_propagation(self, u, v, p, use_softmax=False):
        x_u = self.prop_neighbors[u]
        x_v = self.prop_neighbors[v]

        x = torch.cat((x_u, x_v), -1)

        q_probs = self.q_net_f(x) + self.eps

        θ = self.theta(self.node_features, self.edge_index)
        p_probs = self.relu_func(torch.einsum('k, i->ik', self.eta(), p)) * θ[u] * θ[v]

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        return q_probs, p_probs, self.eta()


class ExactPosteriorECD(nn.Module):
    def __init__(self, num_users, num_communities, neighbors, edge_index, node_features, hyper_params, device=None):
        super().__init__()
        self.link_neighbors, self.prop_neighbors = neighbors
        self.edge_index = edge_index
        self.node_features = node_features

        self.theta = GCN(num_users, num_communities)

        self.phi = PhiGCN(num_users, num_communities)

        r1, r2 = [-1, 1]
        eta_tensor = (r1 - r2) * torch.rand(num_communities) + r2
        self.eta = MyTanhTensor(eta_tensor, coef=1.0)

        self.softmax_func = nn.Softmax(dim=1)
        self.relu_func = nn.ReLU()
        self.eps = 1e-10
        self.s, self.h, self.B = hyper_params

    def forward(self, u, v, p=None):
        if self.p is None:
            q_probs, p_probs, eta = self.forward_link(u, v)
        else:
            q_probs, p_probs, eta = self.forward_prop(u, v, p)

        return q_probs, p_probs, eta

    def forward_link(self, u, v, use_softmax=False):
        θ = self.theta(self.node_features, self.edge_index)
        φ = self.phi(self.node_features, self.edge_index)
        p_probs = (torch.abs(self.eta())) * θ[u] * θ[v] + (1 - torch.abs(self.eta())) * φ[u] * φ[v]

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        # compute exact posterior
        with torch.no_grad():
            alpha = self.h * torch.abs(self.eta()) + self.s * (1 - torch.abs(self.eta()))
            prior = alpha / alpha.sum(dim=-1)
            q_probs = torch.einsum('bc, c->bc', p_probs, prior)
            q_probs = torch.einsum('bc, b->bc', q_probs, 1/q_probs.sum(dim=-1))
        q_probs = q_probs.detach()

        return q_probs, p_probs, self.eta()

    def forward_propagation(self, u, v, p, use_softmax=False):
        θ = self.theta(self.node_features, self.edge_index)
        p_probs = self.relu_func(torch.einsum('k, i->ik', self.eta(), p)) * θ[u] * θ[v]

        if use_softmax:
            p_probs = self.softmax_func(p_probs)
        else:
            p_probs += self.eps

        with torch.no_grad():
            alpha = self.B * torch.abs(self.eta())
            prior = alpha / alpha.sum(dim=-1)
            q_probs = torch.einsum('bc, c->bc', p_probs, prior)
            q_probs = torch.einsum('bc, b->bc', q_probs, 1/q_probs.sum(dim=-1))
        q_probs = q_probs.detach()

        return q_probs, p_probs, self.eta()
