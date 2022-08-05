import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm.autonotebook import tqdm

import logging

DEFAULT_MAX_ALLOWED_POLARIZATION = 1.0
DEFAULT_EC_DECAY = 1.0
DEFAULT_EC_BETA = 1.  # 0.01
DEFAULT_SOCIAL_BETA = .1  # 0.01
DEFAULT_SOCIAL_PRIOR_SIZE = 30.
DEFAULT_EC_PRIOR_SIZE = 16.
DEFAULT_TYPE_FUZZYNESS = 0.05
DEFAULT_PROP_PRIOR_SIZE = 5


def generate_theta_and_phi(N, η,
                           ec_beta=DEFAULT_EC_BETA,
                           social_beta=DEFAULT_SOCIAL_BETA,
                           ):
    ε = 1E-10
    # left pole population
    left_pole_size = N // 3
    is_ec = np.abs(η)
    is_ec[η > 0] = ε
    β1 = is_ec * ec_beta + ε
    θ1 = np.random.dirichlet(β1, size=[left_pole_size])
    # right pole population
    right_pole_size = N // 3
    is_ec = np.abs(η) ** ec_decay
    is_ec[η < 0] = ε
    β2 = is_ec * ec_beta + ε
    θ2 = np.random.dirichlet(β2, size=[right_pole_size])
    # neutral population
    neutral_pole_size = N - (left_pole_size + right_pole_size)
    is_social = 1 - np.abs(η)
    is_social[np.isclose(np.abs(η), [1.])] = ε
    β3 = is_social * ec_beta + ε
    θ3 = np.random.dirichlet(β3, size=[neutral_pole_size])
    # joining the two thetas
    θ = np.vstack([θ1, θ2, θ3])
    # generating phi
    # is_social = (np.abs(η)<1) ** ec_decay
    # β_phi = is_social * social_beta + ε
    φ = []
    for i in range(N):
        φ.append(np.random.dirichlet(θ[i] + ε))
    φ = np.vstack(φ)

    # one-hot vector
    old_theta = θ
    θ = np.zeros(shape=old_theta.shape)
    θ[range(θ.shape[0]), old_theta.argmax(axis=1)] = 1

    θ = np.einsum('nc, n -> nc', θ, 1 / θ.sum(axis=1))
    # φ = np.einsum('nc, n -> nc', φ, 1 / φ.sum(axis=1))
    return θ, φ


def generate_theta_and_phi(N, η,
                           ec_beta=DEFAULT_EC_BETA,
                           social_beta=DEFAULT_SOCIAL_BETA,
                           ):
    ε = 1E-10
    # left pole population
    left_pole_size = N // 3
    is_ec = np.abs(η)
    is_ec[η > 0] = ε
    β1 = is_ec * ec_beta + ε
    θ1 = np.random.dirichlet(β1, size=[left_pole_size])
    # right pole population
    right_pole_size = N // 3
    is_ec = np.abs(η)
    is_ec[η < 0] = ε
    β2 = is_ec * ec_beta + ε
    θ2 = np.random.dirichlet(β2, size=[right_pole_size])
    # neutral population
    neutral_pole_size = N - (left_pole_size + right_pole_size)
    is_social = 1 - np.abs(η)
    β3 = is_social * social_beta + ε
    θ3 = np.random.dirichlet(β3, size=[neutral_pole_size])
    # joining the two thetas
    θ = np.vstack([θ1, θ2, θ3])
    # generating phi
    φ = []
    for i in range(N):
        φ.append(np.random.dirichlet((θ[i] + ε)) * (1 - np.abs(η)))
    φ = np.vstack(φ)
    # normalize theta to obtain a distribution
    θ = np.einsum('nc, n -> nc', θ, 1 / θ.sum(axis=1))
    return θ, φ


class GenerativeModel:
    def __init__(self, N, K=None, M=None, eta=None,
                 social_prior_size=DEFAULT_SOCIAL_PRIOR_SIZE,
                 ec_prior_size=DEFAULT_EC_PRIOR_SIZE,
                 prop_prior_size=DEFAULT_PROP_PRIOR_SIZE,
                 type_fuzzyness=DEFAULT_TYPE_FUZZYNESS,
                 **theta_params
                 ):
        if eta is None:
            self.N, self.K = N, K
            signs = np.sign(np.random.random(self.K) - .5)
            self.η = np.random.beta(type_fuzzyness, type_fuzzyness, self.K) * signs
        else:
            self.η = np.array(eta)
            self.N, self.K = N, len(eta)
            assert K is None or K == self.K

        self.θ, self.φ = generate_theta_and_phi(self.N, self.η, **theta_params)

        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.N))

        self.links = None
        self.pos = None
        self.polarities = None
        self.item2prop = []
        self.degrees = None

        self.π_l = social_prior_size * (1. - np.abs(self.η)) + ec_prior_size * np.abs(self.η)
        self.π_l = self.π_l / np.sum(self.π_l)
        self.π_p = prop_prior_size * np.abs(self.η)
        self.π_p = self.π_p / np.sum(self.π_p)

        if M is not None:
            self.generate_links(M)

        self.mixed = np.einsum('c, nc->nc', abs(self.η), self.θ) + np.einsum('c, nc->nc', 1 - abs(self.η), self.φ)
        self.comm_membership = self.θ

    def set_membership(self, method='theta'):
        if method == 'theta':
            self.comm_membership = self.θ
        elif method == 'phi':
            self.comm_membership = self.φ
        else:
            self.comm_membership = self.mixed

    def compute_node_polarity(self):
        return np.einsum('uk,k->u', self.θ, self.η) / np.sum(self.θ, 1)

    def generate_links(self, n_links):
        self.links = []

        for _ in tqdm(range(n_links), desc="Generating links"):
            # Pick a community according to pre-defined prior
            c = np.random.choice(np.arange(self.K), p=self.π_l)
            # toss a coin to decide if the link is echo-chamber or not
            if np.random.random() < abs(self.η[c]):
                prob_nodes = self.θ[:, c]
            else:
                prob_nodes = self.φ[:, c]
            u, v = np.random.choice(np.arange(self.N), p=(prob_nodes / np.sum(prob_nodes)), size=2)
            self.links.append((u, v))
            # should we check that u != v and that u, v not in links?

        self.G.add_edges_from(self.links)

        # return largest connected component
        retain_lcc = False
        if retain_lcc:
            lcc = list(max(nx.weakly_connected_components(self.G), key=len))
            self.G = self.G.subgraph(lcc)
            self.θ = self.θ[lcc]
            self.φ = self.φ[lcc]
            self.N = len(lcc)
            mapping = {elem: idx for idx, elem in enumerate(lcc)}
            self.G = nx.relabel_nodes(self.G, mapping)

    def _choose_starter(self, p):
        # return np.random.randint(0, self.N)
        # if self.degrees is None:
        #     self.degrees = np.array([self.G.degree(u) for u in range(self.N)])
        # prob_per_node = 1e-3 + (self.θ[:, c])  # * self.degrees)
        # return np.random.choice(np.arange(self.N), p=(prob_per_node / np.sum(prob_per_node)))
        if self.degrees is None:
            self.degrees = np.array([self.G.degree(u) for u in range(self.N)])
        prob_per_community = np.maximum(0, self.η * p)
        prob_per_node = 0.001 + ((self.θ @ prob_per_community) * self.degrees)
        return np.random.choice(np.arange(self.N), p=(prob_per_node / np.sum(prob_per_node)))

    def generate_propagations(self, n_propagations,
                              zipf_exponent=2.0, item_moderation=0.25, discard_below=4, discard_above='N'):

        if discard_above == 'N':
            discard_above = self.N
        elif discard_above is None:
            discard_above = np.inf

        discard_below = discard_below or -1.

        self.polarities = (
                np.random.beta(item_moderation, item_moderation, size=n_propagations) * 2. - 1.)
        self.item2prop = []
        for i in tqdm(range(n_propagations), desc="Generating propagations"):
            p = self.polarities[i]
            size = -np.inf
            while size < discard_below or size > discard_above:
                size = 1 + min(self.N, np.random.zipf(zipf_exponent))
            self.generate_propagation(p, size)

    '''
    def generate_propagation(self, p, size):
        spreaders = None
        prop = []
        while spreaders is None or len(spreaders) == 1:
            prob_per_community = np.maximum(0, self.η * p)  # np.maximum(0, self.π_p*(self.η/abs(self.η))*(p/abs(p)))
            if np.all(prob_per_community == 0.):
                break

            c = np.random.choice(np.arange(self.K),
                                 p=(prob_per_community / np.sum(prob_per_community)))

            starter_node = self._choose_starter(p)
            spreaders = {starter_node}
            exposed = set(self.G.successors(starter_node)) - {starter_node}
            prop = [starter_node]

            for _ in range(size):
                exposed_list = list(exposed)
                prob_per_node = self.θ[exposed_list, c]
                sum_of_prob = np.sum(prob_per_node)
                if not sum_of_prob > 0:
                    logging.debug(f"Stopped propagation at {len(spreaders)}/{size}. sum_of_prob: {sum_of_prob}")
                    break
                u = np.random.choice(exposed_list, p=(prob_per_node / sum_of_prob))
                prop.append(u)

                spreaders.add(u)
                exposed |= set(self.G.successors(u))
                exposed -= spreaders
                if not exposed:
                    logging.debug(f"Stopped propagation at {len(spreaders)}/{size}. {len(exposed)} exposed")
                    break

        self.item2prop.append(prop)
        return len(self.item2prop) - 1
    '''

    def generate_propagation(self, p, size):
        spreaders = None
        prop = []
        while spreaders is None or len(spreaders) == 1:
            u = self._choose_starter(p)
            spreaders = {u}
            exposed = set(self.G.successors(u)) - {u}
            prop = [u]

            prob_per_community = np.maximum(0, self.η * p)
            if np.all(prob_per_community == 0.):
                break

            c = np.random.choice(np.arange(self.K),
                                 p=(prob_per_community / np.sum(prob_per_community)))

            for _ in range(size):
                exposed_list = list(exposed)
                prob_per_node = self.θ[exposed_list, c]
                sum_of_prob = np.sum(prob_per_node)
                if not sum_of_prob > 0:
                    logging.debug(f"Stopped propagation at {len(spreaders)}/{size}. sum_of_prob: {sum_of_prob}")
                    break
                u = np.random.choice(exposed_list, p=(prob_per_node / sum_of_prob))
                prop.append(u)

                spreaders.add(u)
                exposed |= set(self.G.successors(u))
                exposed -= spreaders
                if not exposed:
                    logging.debug(f"Stopped propagation at {len(spreaders)}/{size}. {len(exposed)} exposed")
                    break

        self.item2prop.append(prop)
        return len(self.item2prop) - 1

    def _pos(self):
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, seed=123, k=.25, scale=3)

        # for nonconnected in list(nx.connected_components(self.G))[1:]:
        #     for u in nonconnected:
        #         self.pos[u] = np.array([0, 0])

        return self.pos

    def _figure(self, node_color, cmap, node_size=70, alpha_nodes=1., alpha_edges=0.25, white_background=False):
        if white_background:
            bg_color, nodes_kw, edge_color = 'white', {'edgecolors': 'black'}, 'dimgray'
        else:
            bg_color, nodes_kw, edge_color = 'dimgray', {}, 'white'

        plt.figure(figsize=(8, 8), facecolor=bg_color)
        plt.axis('off')
        nx.draw_networkx_nodes(self.G, self._pos(),
                               node_size=node_size, alpha=alpha_nodes,
                               node_color=node_color, cmap=cmap, **nodes_kw)
        nx.draw_networkx_edges(self.G, self._pos(), alpha=alpha_edges, edge_color=edge_color)
        return edge_color

    def draw(self, **kwargs):
        self._figure(self.compute_node_polarity(), 'bwr', **kwargs)

    def draw_communities(self, communities=None, eta_color=False, **kwargs):
        if eta_color:
            node_color = self.η[
                np.argmax(self.comm_membership if communities is None else self.comm_membership[:, communities],
                          axis=1)]
            self._figure(node_color, 'coolwarm', **kwargs)
        else:
            node_color = np.argmax(
                self.comm_membership if communities is None else self.comm_membership[:, communities], axis=1)
            self._figure(node_color, 'prism', **kwargs)

    def draw_community(self, k, edge_color="w", **kwargs):
        nodes_in_comm = np.where(self.comm_membership[:, k] > np.percentile(self.comm_membership[:, k], 90))[0]
        pos_comm = {u: p for u, p in self._pos().items() if u in nodes_in_comm}
        edge_color = self._figure(self.comm_membership[:, k], 'hot', alpha_edges=0.05, **kwargs)
        nx.draw_networkx_edges(self.G.subgraph(nodes_in_comm), pos_comm,
                               alpha=1., width=1., edge_color=edge_color)
        plt.title("Community %d. Echo chamber: %.2f" % (k, self.η[k]))

    def draw_propagation(self, i, width_prop_edges=1., alpha_other_edges=0.05, **kwargs):
        alpha_nodes = [1. if u in self.item2prop[i] else .25 for u in range(self.N)]
        edge_color = self._figure(self.compute_node_polarity(), 'bwr',
                                  alpha_nodes=alpha_nodes, alpha_edges=alpha_other_edges, **kwargs)
        pos_comm = {u: p for u, p in self._pos().items() if u in self.item2prop[i]}

        prop_edges = list()
        for j, u in enumerate(self.item2prop[i][1:], 1):
            active = set(self.item2prop[i][:j])
            prop_edges += [
                (v, u) for v in (set(self.G.predecessors(u)) & active)
            ]

        nx.draw_networkx_edges(self.G.subgraph(self.item2prop[i]).edge_subgraph(prop_edges),
                               pos_comm, alpha=1., width=width_prop_edges,
                               edge_color=edge_color)
        # plt.title("Item %d. Item polarity: %.2f" % (i, self.polarities[i]))
