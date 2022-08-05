import torch
import numpy as np
import networkx as nx
import pandas as pd
import random
import itertools
import models
import utils
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax as numpy_softmax
from scipy.stats import pearsonr
from plottify import autosize
from data_loaders import Dataset, PairwiseDataset
from annealer import ConstantAnnealer, CyclicAnnealer


def set_seed(seed):
    if seed is None:
        seed = 12121995
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten(t):
    return [item for sublist in t for item in sublist]


def build_dataset(edge_list, propagations, polarities, oversampling_minority=True):
    prop_edge_list = list(map(lambda x: list(itertools.permutations(x, 2)), propagations))
    prop_polarities = [[polarities[idx]] * len(prop_edge_list[idx]) for idx in range(len(prop_edge_list))]
    prop_edge_list = np.array(flatten(prop_edge_list))
    prop_polarities = np.array(flatten(prop_polarities))

    if oversampling_minority:
        num_props = prop_edge_list.shape[0]
        num_edges = edge_list.shape[0]

        if num_props > num_edges:
            edge_idxs = range(num_edges)
            sampled_edges = np.random.choice(edge_idxs, size=num_props - num_edges, replace=True)
            edge_list = np.vstack([edge_list, edge_list[sampled_edges]])
        elif num_edges > num_props:
            edge_idxs = range(num_props)
            sampled_edges = np.random.choice(edge_idxs, size=num_edges - num_props, replace=True)
            prop_edge_list = np.vstack([prop_edge_list, prop_edge_list[sampled_edges]])
            prop_polarities = np.hstack([prop_polarities, prop_polarities[sampled_edges]])

    numpy_u = np.hstack([edge_list[:, 0], prop_edge_list[:, 0]])
    numpy_v = np.hstack([edge_list[:, 1], prop_edge_list[:, 1]])
    numpy_p = np.hstack([[np.nan] * edge_list.shape[0], prop_polarities])
    u = torch.tensor(numpy_u)
    v = torch.tensor(numpy_v)
    p = torch.tensor(numpy_p)
    return u, v, p


def build_pairwise_dataset(edge_list, propagations, polarities, oversampling_minority=True):
    prop_edge_list = list(map(lambda x: list(itertools.permutations(x, 2)), propagations))
    prop_polarities = [[polarities[idx]] * len(prop_edge_list[idx]) for idx in range(len(prop_edge_list))]
    prop_edge_list = np.array(flatten(prop_edge_list))
    prop_polarities = np.array(flatten(prop_polarities))

    prop_id_list = list(
        map(lambda x: len(propagations[x]) * (len(propagations[x]) - 1) * [x], range(len(propagations))))
    prop_id_list = np.array(flatten(prop_id_list))

    all_users = set(range(edge_list.max() + 1))
    negatives_by_prop = list(map(lambda idx: list(all_users - set(propagations[idx])), range(len(propagations))))
    negatives_nodes = np.array(list(map(lambda x: np.random.choice(negatives_by_prop[x]), prop_id_list)))

    negatives_by_link = list(map(lambda idx: list(all_users - set(edge_list[edge_list[:, 0] == idx][:, 1] + [idx])),
                                 range(edge_list.max() + 1)))
    negatives_nodes_by_link = np.array(
        list(map(lambda x: np.random.choice(negatives_by_link[x]), prop_edge_list[:, 0])))

    if oversampling_minority:
        num_props = prop_edge_list.shape[0]
        num_edges = edge_list.shape[0]

        if num_props > num_edges:
            edge_idxs = range(num_edges)
            sampled_edges = np.random.choice(edge_idxs, size=num_props - num_edges, replace=True)
            edge_list = np.vstack([edge_list, edge_list[sampled_edges]])
            negatives_nodes_by_link = np.hstack([negatives_nodes_by_link, negatives_nodes_by_link[sampled_edges]])
        elif num_edges > num_props:
            edge_idxs = range(num_props)
            sampled_edges = np.random.choice(edge_idxs, size=num_edges - num_props, replace=True)
            prop_edge_list = np.vstack([prop_edge_list, prop_edge_list[sampled_edges]])
            prop_polarities = np.vstack([prop_polarities, prop_polarities[sampled_edges]])
            negatives_nodes = np.hstack([negatives_nodes, negatives_nodes[sampled_edges]])

    u_pos = torch.tensor(np.hstack([edge_list[:, 0], prop_edge_list[:, 0]]))
    v_pos = torch.tensor(np.hstack([edge_list[:, 1], prop_edge_list[:, 1]]))
    p_pos = torch.tensor(np.hstack([[np.nan] * edge_list.shape[0], prop_polarities]))
    u_neg = torch.tensor(np.hstack([edge_list[:, 0], prop_edge_list[:, 0]]))
    v_neg = torch.tensor(np.hstack([negatives_nodes_by_link, negatives_nodes]))
    p_neg = torch.tensor(np.hstack([[np.nan] * edge_list.shape[0], prop_polarities]))
    return [u_pos, u_neg], [v_pos, v_neg], [p_pos, p_neg]


def create_dataloader(edge_list, propagations, polarities, oversampling_minority, training_type, cuda_device,
                      batch_size):
    params = {
        'batch_size': batch_size,
        'shuffle': True
    }
    if training_type == models.DEFAULT_TRAINING_TYPE:
        u, v, p = build_dataset(edge_list, propagations, polarities, oversampling_minority)
        y = torch.ones(u.shape[0], device=cuda_device, requires_grad=False)
        dataset = Dataset(u, v, p, y, cuda_device)
        dataloader = torch.utils.data.DataLoader(dataset, **params)
        # Create a validation set by sampling the original dataset
        val_perc = 0.15
        val_idxs = np.random.permutation(range(u.shape[0]))[:int(val_perc * u.shape[0])]
        val_dataset = Dataset(u[val_idxs], v[val_idxs], p[val_idxs], y[val_idxs], cuda_device)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **params)
    elif training_type == models.PAIRWISE_TRAINING_TYPE:
        u, v, p = build_pairwise_dataset(edge_list, propagations, polarities, oversampling_minority)
        dataset = PairwiseDataset(u, v, p, cuda_device)
        dataloader = torch.utils.data.DataLoader(dataset, **params)
        # Create a validation set by sampling the original dataset
        val_perc = 0.15
        val_idxs = np.random.permutation(range(u[0].shape[0]))[:int(val_perc * u[0].shape[0])]
        val_u = [elem[val_idxs] for elem in u]
        val_v = [elem[val_idxs] for elem in v]
        val_p = [elem[val_idxs] for elem in p]
        val_dataset = PairwiseDataset(val_u, val_v, val_p, cuda_device)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **params)
    else:
        raise Exception(f'Training type: {training_type} does not exist.')

    return dataloader, val_dataloader


def train_one_epoch(dataloader, optimizer, ablation, model, criterion, annealer, reweighting=False):
    cum_loss_history = []
    cum_nll_history = []
    cum_kld_history = []
    # Training
    for u, v, p, y in dataloader:
        annealer.update()
        criterion.anneal_param = annealer.anneal_param
        optimizer.zero_grad()

        link_mask = torch.isnan(p)
        link_u, link_v = u[link_mask], v[link_mask]
        prop_u, prop_v, prop_p = u[~link_mask], v[~link_mask], p[~link_mask]

        # Link
        if ablation != models.LINK_ABLATION:
            q_probs, p_probs, eta = model.forward_link(link_u, link_v, True)
            nll, kld = criterion(q_probs, p_probs, eta, y[link_mask], is_link=True)
            nll = nll.sum()
            kld = kld.sum()
            weight = 1.0
            if reweighting:
                # balancing the contribution based on the relative number of links on the total
                num_props, num_links = ((~link_mask).sum()), link_mask.sum()
                weight = num_props / num_links
            loss = weight*(nll+kld)
        # Propagations
        if ablation != models.PROP_ABLATION:
            q_probs, p_probs, eta = model.forward_propagation(prop_u, prop_v, prop_p, True)
            if ablation != models.LINK_ABLATION:
                prop_nll, prop_kld = criterion(q_probs, p_probs, eta, y[~link_mask], is_link=False, p=prop_p)
                nll += prop_nll.sum()
                kld += prop_kld.sum()
                loss += (prop_nll.sum()+prop_kld.sum())
            else:
                nll, kld = criterion(q_probs, p_probs, eta, y[~link_mask], is_link=False, p=prop_p)
                nll = nll.sum()
                kld = kld.sum()
                loss = nll + kld

        loss.backward()
        optimizer.step()

        cum_loss_history.append(loss.item())
        cum_nll_history.append(nll.item())
        cum_kld_history.append(kld.item())

    return model, cum_loss_history, cum_nll_history, cum_kld_history


def eval_one_epoch(val_dataloader, model, criterion, ablation):
    # Validation Step
    with torch.no_grad():
        cum_loss_history = []
        cum_nll_history = []
        cum_kld_history = []
        for u, v, p, y in val_dataloader:
            link_mask = torch.isnan(p)
            link_u, link_v = u[link_mask], v[link_mask]
            prop_u, prop_v, prop_p = u[~link_mask], v[~link_mask], p[~link_mask]

            # Link
            if ablation != models.LINK_ABLATION:
                q_probs, p_probs, eta = model.forward_link(link_u, link_v, True)
                nll, kld = criterion(q_probs, p_probs, eta, y[link_mask], is_link=True)
                nll = nll.sum()
                kld = kld.sum()
                loss = nll+kld
            # Propagations
            if ablation != models.PROP_ABLATION:
                q_probs, p_probs, eta = model.forward_propagation(prop_u, prop_v, prop_p, True)
                if ablation != models.LINK_ABLATION:
                    prop_nll, prop_kld = criterion(q_probs, p_probs, eta, y[~link_mask], is_link=False, p=prop_p)
                    nll += prop_nll.sum()
                    kld += prop_kld.sum()
                    loss += (prop_nll.sum() + prop_kld.sum())
                else:
                    nll, kld = criterion(q_probs, p_probs, eta, y[~link_mask], is_link=False, p=prop_p)
                    nll = nll.sum()
                    kld = kld.sum()
                    loss = nll+kld

            cum_loss_history.append(loss.item())
            cum_nll_history.append(nll.item())
            cum_kld_history.append(kld.item())

    return cum_loss_history, cum_nll_history, cum_kld_history


def eval_stance_prediction(link_adj, model, cuda_device, data_dir, propagations, polarities, BEST_VAL_METRIC,
                           interim_data_path, val_stance_prediction_trend, logger, estimation_methods, model_type,
                           user_mapping=None):
    G = nx.from_numpy_array(link_adj, create_using=nx.DiGraph)
    best_theta = 0.5
    with torch.no_grad():
        for estimation_method in estimation_methods:
            estimated_polarities = utils.get_estimated_polarities(model=model,
                                                                  link_adj=link_adj,
                                                                  cuda_device=cuda_device,
                                                                  method=estimation_method,
                                                                  model_type=model_type)

            manual_stances = utils.open_manual_stances(path=data_dir / 'manual_stances.tsv',
                                                       include_username=user_mapping is not None)
            if user_mapping is not None:
                manual_stances['user'] = manual_stances['username'].apply(lambda x: user_mapping[x])
            user_polarities_by_prop = utils.get_user_polarities_by_prop(G, propagations, polarities)
            test_df = utils.build_data_for_stance_prediction(manual_stances, estimated_polarities,
                                                             user_polarities_by_prop)
            _, roc_auc, best_th = utils.plot_roc_auc(test_df)
            if estimation_method == 'theta':
                best_theta = best_th
            logger.info(f'Estimation Method: {estimation_method} --- Val AUC: {round(roc_auc, 4)}')
            if roc_auc > BEST_VAL_METRIC:
                torch.save(model.state_dict(), interim_data_path / 'best_val_AUC_model.pt')
                BEST_VAL_METRIC = roc_auc
            val_stance_prediction_trend[estimation_method].append(roc_auc)

    return val_stance_prediction_trend, BEST_VAL_METRIC, best_theta


def generate_prop_adj(propagations, num_users):
    prop_adj = np.zeros(shape=(num_users, num_users), dtype=int)
    prop_edge_list = list(map(lambda x: list(itertools.permutations(x, 2)), propagations))
    prop_edge_list = np.array(flatten(prop_edge_list))
    prop_adj[prop_edge_list[:, 0], prop_edge_list[:, 1]] = 1
    return prop_adj


def open_manual_stances(path, include_username=False):
    selected_columns = ["user", "stance", "stance (A2)"]
    if include_username:
        selected_columns.append("username")
    manual_stances = pd.read_csv(path, sep="\t")[selected_columns]

    # mantain only concordant stances between the two annotators
    concordant_pairs = manual_stances.apply(lambda x: x[1] == x[2], axis=1)
    manual_stances = manual_stances[concordant_pairs]

    # excluding neutral users
    manual_stances = manual_stances[manual_stances.stance != 0]
    selected_columns = ['user', 'stance']
    if include_username:
        selected_columns[0:0] = ['username']
    manual_stances = manual_stances[selected_columns]

    return manual_stances


def get_user_polarities_by_prop(G, propagations, polarities):
    cum_prop_polarities = np.zeros(G.number_of_nodes())
    num_prop = np.zeros(G.number_of_nodes())
    for propagation_id in range(len(propagations)):
        for user in propagations[propagation_id]:
            cum_prop_polarities[user] += polarities[propagation_id]
            num_prop[user] += 1
    user_polarities_by_prop = cum_prop_polarities / num_prop
    user_polarities_by_prop[np.isnan(user_polarities_by_prop)] = .5  # neutral prediction
    return user_polarities_by_prop


def build_data_for_stance_prediction(manual_stances, estimated_polarities, user_polarities_by_prop):
    ground_truth = []
    estimated_polarity = []
    baseline_estimated_polarity = []
    for user in manual_stances.user.values:
        ground_truth.append(manual_stances[manual_stances.user == user].stance.item())
        estimated_polarity.append(estimated_polarities[user])
        baseline_estimated_polarity.append(user_polarities_by_prop[user])

    return pd.DataFrame(data={"ground_truth": ground_truth, "estimated_polarity": estimated_polarity,
                              'baseline_polarity': baseline_estimated_polarity})


def plot_roc_auc(test_df):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scaler = MinMaxScaler()
    my_fpr, my_tpr, my_th = roc_curve(test_df.ground_truth.astype(str),
                                      scaler.fit_transform(test_df.estimated_polarity.values.reshape(-1, 1)),
                                      pos_label="1")
    my_roc_auc = auc(my_fpr, my_tpr)
    if my_roc_auc < 0.5:
        my_fpr, my_tpr, my_th = roc_curve(test_df.ground_truth.astype(str),
                                          scaler.fit_transform(test_df.estimated_polarity.values.reshape(-1, 1)),
                                          pos_label="-1")
        my_roc_auc = auc(my_fpr, my_tpr)

    lw = 2
    ax.plot(
        my_fpr,
        my_tpr,
        lw=lw,
        label="ROC curve (area = %0.2f)" % my_roc_auc,
    )
    ax.plot([0, 1], [0, 1], lw=lw, linestyle="--")

    scaler = MinMaxScaler()
    fpr, tpr, _ = roc_curve(test_df.ground_truth.astype(str),
                            scaler.fit_transform(test_df.baseline_polarity.values.reshape(-1, 1)), pos_label="1")
    roc_auc = auc(fpr, tpr)

    lw = 2
    ax.plot(
        fpr,
        tpr,
        lw=lw,
        label="ROC curve (area = %0.2f) baseline" % roc_auc,
    )
    ax.plot([0, 1], [0, 1], lw=lw, linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    ax.legend(fontsize=16)
    autosize(fig)
    return fig, my_roc_auc, my_th[np.argmax(my_tpr - my_fpr)]


def get_theta(model, model_type, nusers, cuda_device):
    softmax = torch.nn.Softmax(dim=1)
    if model_type not in ['gcn', 'exact-posterior']:
        theta = softmax(model.theta.embedding(torch.tensor(range(nusers), device=cuda_device))).detach().cpu().numpy()
    else:
        theta = model.theta(model.node_features, model.edge_index).detach().cpu().numpy()
    return theta


def get_phi(model, model_type, nusers, cuda_device):
    softmax = torch.nn.Softmax(dim=1)
    if model_type not in ['gcn', 'exact-posterior']:
        phi = softmax(model.phi.embedding(torch.tensor(range(nusers), device=cuda_device))).detach().cpu().numpy()
    else:
        phi = model.phi(model.node_features, model.edge_index).detach().cpu().numpy()
    return phi


def get_estimated_polarities(model, link_adj, cuda_device, method='theta', model_type='sigmoid'):
    with torch.no_grad():
        theta = get_theta(model, model_type, link_adj.shape[0], cuda_device)
        if method in ['phi', 'mixed', 'mixed_alt']:
            phi = get_phi(model, model_type, link_adj.shape[0], cuda_device)
        eta = model.eta().detach().cpu().numpy()

        if method == 'theta':
            estimated_polarities = (eta * theta).sum(axis=1)
        elif method == 'phi':
            estimated_polarities = (eta * phi).sum(axis=1)
        elif method == 'mixed':
            estimated_polarities = (((theta + phi)/2)*eta).sum(axis=1)
            estimated_polarities = np.clip(estimated_polarities, -1., 1.)
        elif method == 'mixed_alt':
            estimated_polarities = (numpy_softmax(theta + phi, axis=1)*eta).sum(axis=1)
            estimated_polarities = np.clip(estimated_polarities, -1., 1.)
        else:
            raise Exception(f'method: {method} is not supported!')

    return estimated_polarities


def modify_graph(G, percentile=95):
    inner_graph = G.copy()
    degree_max = np.percentile(list(dict(inner_graph.degree).values()), percentile)
    hubs = {node for node, degree in inner_graph.degree if degree > degree_max}
    for node in hubs:
        out_edges = list(inner_graph.out_edges(node)).copy()
        inner_graph.remove_edges_from(out_edges)
    return inner_graph


def compute_RWC(polarity_partition, G, th=0):
    G1 = modify_graph(G)

    X = np.where(polarity_partition < th)[0]
    Y = np.where(polarity_partition >= th)[0]
    if len(X) == 0 or len(Y) == 0:
        return None

    nstart_X = {x: (1 if x in X else 0) for x in G1.nodes()}
    pr_X = nx.pagerank(G1, alpha=0.85, nstart=nstart_X, personalization=nstart_X)
    nstart_Y = {x: (1 if x in Y else 0) for x in G1.nodes()}
    pr_Y = nx.pagerank(G1, alpha=0.85, nstart=nstart_Y, personalization=nstart_Y)

    P_XX = sum([pr_X[u] for u in X])
    P_YY = sum([pr_Y[u] for u in Y])
    P_XY = sum([pr_X[u] for u in Y])
    P_YX = sum([pr_Y[u] for u in X])

    RWC = P_XX * P_YY - P_YX * P_XY
    return RWC


def plot_losses(losses, title, log_scale=False, xlabel='Epoch'):
    loss_trend, nll_trend, kld_trend = losses
    fig, ax = plt.subplots(figsize=(8, 6))
    if xlabel == 'Epoch':
        ax.plot(loss_trend, label='cum')
        ax.plot(nll_trend, label='nll')
        ax.plot(kld_trend, label='kld')
    else:
        ax.scatter(range(len(loss_trend)), loss_trend, alpha=0.25, label='cum')
        ax.scatter(range(len(nll_trend)), nll_trend, alpha=0.25, label='nll')
        ax.scatter(range(len(kld_trend)), kld_trend, alpha=0.25, label='kld')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    if log_scale:
        ax.set_yscale('log')
    autosize(fig)
    fig.tight_layout()
    return fig


def plot_val_auc(estimation_methods, val_stance_prediction_trend, xlabel='Epoch'):
    fig, ax = plt.subplots(figsize=(8, 6))
    for estimation_method in estimation_methods:
        ax.plot(val_stance_prediction_trend[estimation_method], label=estimation_method)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ROC-AUC')
    ax.legend()
    autosize(fig)
    fig.tight_layout()
    return fig


def init_annealer(annealing, n_iter):
    if annealing == 'constant':
        annealer = ConstantAnnealer(1.0)
    elif annealing in ['cyclic', 'linear', 'sigmoid']:
        annealer = CyclicAnnealer(0.0, M=4, R=0.5, T=n_iter, policy_type=annealing)
    elif annealing == 'without':
        annealer = ConstantAnnealer(0.0)
    else:
        raise NotImplementedError(f'Annealer {annealing} is not implemented yet')
    return annealer


def mae_func(vec1, vec2):
    return abs(vec1.mean() - vec2.mean())


def compute_ec_stats(x, users_to_prop, users_prop, polarities, npairs=250):
    share_at_least_one_list = []
    jaccard_list = []
    mae_list = []
    cnt = 0
    prop_mask = [users_to_prop[elem].sum() > 0 for elem in x]
    print(f'doing a comm...\n Num Users: {len(x)}')
    if np.sum(prop_mask) <= 1 or np.sum(prop_mask) <= (len(x)-25):
        return [0.], [0.], [0.]
    while cnt < npairs:
        users_pair = np.random.permutation(x[prop_mask])[:2]
        if users_to_prop[users_pair[0]].sum() == 0 or users_to_prop[users_pair[1]].sum() == 0:
            continue
        cnt += 1
        # 1st: users pair activated on at least 1 propagation
        share_at_least_one = np.logical_and(users_to_prop[users_pair[0]], users_to_prop[users_pair[1]]).sum() > 0
        # 2nd: jaccard between the two propagation sets
        jaccard = (np.logical_and(users_to_prop[users_pair[0]], users_to_prop[users_pair[1]]).sum()) / (
            np.logical_or(users_to_prop[users_pair[0]], users_to_prop[users_pair[1]]).sum())
        # 3rd: users activate on propagations with similar stances
        mae_stance = mae_func(polarities[users_prop[users_pair[0]]], polarities[users_prop[users_pair[1]]])
        share_at_least_one_list.append(share_at_least_one)
        jaccard_list.append(jaccard)
        mae_list.append(mae_stance)
    return share_at_least_one_list, jaccard_list, mae_list


def echo_chamber_analysis(model, link_adj, cuda_device, propagations, polarities, NPAIRS=20000, model_type='sigmoid'):
    with torch.no_grad():
        θ = get_theta(model, model_type, link_adj.shape[0], cuda_device)
        η = model.eta().detach().cpu().numpy()
    comm_memberships = θ.argmax(axis=1)
    num_users_by_comm = map(lambda comm_id: (comm_memberships == comm_id).sum(), range(η.shape[0]))
    comm_stats = pd.DataFrame(data={'comm_id': range(η.shape[0]),
                                    'eta': η,
                                    'nusers': num_users_by_comm,
                                    }).sort_values(by='nusers', ascending=False)

    users_prop = {u_idx: [] for u_idx in range(link_adj.shape[0])}
    for idx in range(len(propagations)):
        propagation = propagations[idx]
        for u_idx in propagation:
            users_prop[u_idx].append(idx)
    users_to_prop = np.zeros(shape=(link_adj.shape[0], len(propagations)), dtype=int)
    for propagation_idx in range(len(propagations)):
        for u_idx in propagations[propagation_idx]:
            users_to_prop[u_idx, propagation_idx] = 1
    x = np.array(list(range(link_adj.shape[0])))
    share_list, jaccard_list, mae_list = [], [], []
    for comm_idx in comm_stats.comm_id:
        comm_users = x[comm_memberships == comm_idx]
        if len(comm_users) < 10:
            share_list.append(None)
            jaccard_list.append(None)
            mae_list.append(None)
        else:
            share, jaccard, mae = compute_ec_stats(comm_users, users_to_prop, users_prop, polarities, npairs=NPAIRS)
            share_list.append(np.mean(share))
            jaccard_list.append(np.mean(jaccard))
            mae_list.append(np.mean(mae))

    comm_stats['share_index'] = share_list
    comm_stats['jaccard_index'] = jaccard_list
    comm_stats['mae_index'] = mae_list
    comm_stats['abs_eta'] = comm_stats['eta'].apply(lambda eta_val: abs(eta_val))

    valid_idxs = comm_stats.nusers > 20
    if valid_idxs.sum() < 2:
        share_corr = jaccard_corr = mae_corr = 1
    else:
        share_corr = round(pearsonr(comm_stats['abs_eta'][valid_idxs], comm_stats['share_index'][valid_idxs])[0], 4)
        jaccard_corr = round(pearsonr(comm_stats['abs_eta'][valid_idxs], comm_stats['jaccard_index'][valid_idxs])[0], 4)
        mae_corr = round(pearsonr(comm_stats['abs_eta'][valid_idxs], comm_stats['mae_index'][valid_idxs])[0], 4)

    return share_corr, jaccard_corr, mae_corr, comm_stats


def plot_target_reco(trends):
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(trends[0], label='θ')
    ax.plot(trends[1], label='φ')
    ax.plot(trends[2], label='η')
    ax.set_xlabel('epochs')
    ax.set_ylabel(r'$|v - \hat{v}|$')
    ax.legend()
    autosize(fig)
    fig.tight_layout()
    return fig


def get_our_partition(θ, num_communities):
    comm_memberships = θ.argmax(axis=1)
    partition_as_sequence_of_sets = [[] for _ in range(num_communities)]
    for node_id in range(len(comm_memberships)):
        comm_id = comm_memberships[node_id]
        partition_as_sequence_of_sets[comm_id].append(node_id)

    return partition_as_sequence_of_sets


def plot_purity_conductance(model, link_adj, prop_adj, cuda_device, num_communities, best_th, model_type):
    def __cal_conductance(G, cluster):
        """cluster: a list of node id that forms a algorithms. Data type of cluster is given by numpy array
        Calculate the conductance of the cut A and complement of A.
        """

        assert (
                type(cluster) == np.ndarray
        ), "The given algorithms members is not a numpy array"

        temp = G[cluster, :]
        subgraph = temp[:, cluster]
        cutsize = temp.sum() - subgraph.sum()
        denominator = min(temp.sum(), G.sum() - temp.sum())
        conductance = cutsize / denominator if denominator > 0 else 1

        return conductance

    def get_category(polarity, th=0.25):
        if polarity <= th:
            return 'pro'
        elif polarity > th:
            return 'anti'

    def get_purity(comm, node_labels, th=0.25):
        party_stat = {"pro": 0, "anti": 0, "neutral": 0}
        for node_id in comm:
            party_stat[get_category(node_labels[node_id], th)] += 1

        num = max([party_stat[stance] for stance in party_stat])
        return num / (party_stat["pro"] + party_stat["anti"] + party_stat['neutral'])

    with torch.no_grad():
        θ = get_theta(model, model_type, link_adj.shape[0], cuda_device)
        η = model.eta().detach().cpu().numpy()

    scaler = MinMaxScaler()
    node_polarities = (η * θ).sum(axis=1)
    rescaled_node_polarities = scaler.fit_transform(node_polarities.reshape(-1, 1))

    our_partition = get_our_partition(θ, num_communities)
    propagation_graph_flag = True
    our_purity, our_conductance, our_size = [], [], []

    for idx in range(len(our_partition)):
        if len(our_partition[idx]) == 0:
            our_purity.append(0.)
            our_conductance.append(0.)
            our_size.append(0)
        else:
            our_purity.append(get_purity(our_partition[idx], rescaled_node_polarities, th=best_th))
            if propagation_graph_flag:
                our_conductance.append(__cal_conductance(prop_adj, np.array(our_partition[idx])))
            else:
                our_conductance.append(__cal_conductance(link_adj, np.array(our_partition[idx])))
            our_size.append(len(our_partition[idx]))

    our_purity = np.array(our_purity)
    our_conductance = np.array(our_conductance)
    our_size = np.array(our_size)

    fig, ax = plt.subplots(figsize=(8, 8))

    valid_idxs = our_size > 25
    ax.scatter(our_conductance[valid_idxs],
               our_purity[valid_idxs],
               alpha=1.0, c=η[valid_idxs], cmap='RdBu', vmin=-1.0, vmax=1.0)

    ax.axhline(0.7, linestyle="--", alpha=.5)
    ax.axvline(0.5, linestyle="--", alpha=.5)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Conductance")
    ax.set_ylabel("Purity")
    ax.set_title("Our")

    norm = plt.Normalize(-1, 1.)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    ax.figure.colorbar(sm, label="Polarity")

    autosize(fig)
    fig.tight_layout()
    return fig


def plot_ec_indexes(val_ec_trend):
    fig, ax = plt.subplots(figsize=(8, 6))

    for ec_index in val_ec_trend:
        if ec_index == 'mae':
            ax.plot(val_ec_trend[ec_index], label=ec_index, linestyle='--')
        else:
            ax.plot(val_ec_trend[ec_index], label=ec_index)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Echo-Chamber')
    ax.legend()
    autosize(fig)
    fig.tight_layout()
    return fig
