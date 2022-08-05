from losses import EchoChamberLoss

import os
import math
import shutil
import uuid
import sys
import logging
import mlflow
import pathlib
import torch
import networkx as nx
import numpy as np
import pickle
import argparse
import seaborn as sns
import models
import utils

sns.set()


def run_experiment(
        dataset_name,
        link_adj,
        prop_adj,
        edge_list,
        polarities,
        propagations,
        u2idx,
        num_communities,
        ec_prior_size=models.EC_PRIOR_SIZE,
        social_prior_size=models.SOCIAL_PRIOR_SIZE,
        propagation_prior_size=models.PROPAGATION_PRIOR_SIZE,
        removed_links=[],
        num_props_removed=0,
        stance_detection_exp=False,
        ablation=models.DEFAULT_ABLATION,
        training_type=models.DEFAULT_TRAINING_TYPE,
        model_type='sigmoid',
        # Learning params
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        oversampling_minority=True,
        reweighting=False,
        annealing='cyclic',
        seed=12121995,
        device=""
):
    mlflow.log_param('seed', seed)
    mlflow.log_param('ablation', ablation)
    mlflow.log_param('training_type', training_type)
    mlflow.log_param('link_removal', len(removed_links))
    mlflow.log_param('prop_removal', num_props_removed)
    mlflow.log_param('stance_detection_exp', stance_detection_exp)
    mlflow.log_param('model_type', model_type)
    mlflow.log_param('dataset', dataset_name)
    mlflow.log_param('num_communities', num_communities)
    mlflow.log_param('ec_prior_size', ec_prior_size)
    mlflow.log_param('social_prior_size', social_prior_size)
    mlflow.log_param('propagation_prior_size', propagation_prior_size)
    mlflow.log_param('epochs', epochs)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('oversampling_minority', oversampling_minority)
    mlflow.log_param('reweighting', reweighting)
    mlflow.log_param('annealing', annealing)

    # Creating scripts to host run-specific files
    my_run_id = uuid.uuid4()
    curr_dir = pathlib.Path.cwd()
    data_dir = curr_dir.parent / 'data' / 'raw' / dataset_name
    interim_data_path = curr_dir.parent / 'data' / 'interim' / f"{my_run_id}"
    if not os.path.exists(interim_data_path):
        os.makedirs(interim_data_path)

    # Set seed for reproducibility
    utils.set_seed(seed)

    # Set cuda visible and set cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    if device != '':
        torch.cuda.set_device(int(device))
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.log_param("cuda", cuda_device)

    # Setting logger. Must save on a .txt file AND print on console terminal
    file_handler = logging.FileHandler(filename=interim_data_path / 'log.txt')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(handlers=handlers, format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger("my logger")

    # if this experiment comprises edges/prop removal then save the removed edges/prop
    if len(removed_links) > 0:
        with open(interim_data_path / 'removed_links.pkl', 'wb') as f:
            pickle.dump(removed_links, f)
        mlflow.log_artifact(interim_data_path / 'removed_links.pkl')
    if num_props_removed > 0:
        with open(interim_data_path / 'props.pkl', 'wb') as f:
            pickle.dump(propagations, f)
        mlflow.log_artifact(interim_data_path / 'props.pkl')

    # Getting pytorch representation of social and interaction graph adjacency matrix
    torch_link_adj = torch.tensor(link_adj, device=cuda_device, requires_grad=False).float()
    torch_prop_adj = torch.tensor(prop_adj, device=cuda_device, requires_grad=False).float()

    # Creating the model for Echo Chamber Detection
    if model_type == 'sigmoid':
        model = models.ECD(link_adj.shape[0], num_communities, [torch_link_adj, torch_prop_adj], cuda_device)
    elif model_type == 'softmax':
        model = models.AltECD(link_adj.shape[0], num_communities, [torch_link_adj, torch_prop_adj], cuda_device)
    elif model_type == 'exact-posterior':
        torch_edge_list = torch.tensor(edge_list.T, dtype=torch.long, requires_grad=False).to(cuda_device)
        node_features = torch.eye(link_adj.shape[0], dtype=torch.float, requires_grad=False).to(cuda_device)
        assert torch_edge_list.min() >= 0
        assert torch_edge_list.max() < node_features.size(0)
        model = models.ExactPosteriorECD(link_adj.shape[0], num_communities, [torch_link_adj, torch_prop_adj],
                                            edge_index=torch_edge_list, node_features=node_features,
                                            hyper_params=[ec_prior_size, social_prior_size, propagation_prior_size],
                                            device=cuda_device)
    elif model_type == 'post-softmax':
        model = models.PostSoftmaxECD(link_adj.shape[0], num_communities, [torch_link_adj, torch_prop_adj], cuda_device)
    elif model_type == 'gcn':
        torch_edge_list = torch.tensor(edge_list.T, dtype=torch.long, requires_grad=False).to(cuda_device)
        node_features = torch.eye(link_adj.shape[0], dtype=torch.float, requires_grad=False).to(cuda_device)
        assert torch_edge_list.min() >= 0
        assert torch_edge_list.max() < node_features.size(0)
        model = models.GCN_ECD(link_adj.shape[0], num_communities, [torch_link_adj, torch_prop_adj],
                               edge_index=torch_edge_list, node_features=node_features, device=cuda_device)
    else:
        raise NotImplementedError(f'Model type: {model_type} not implemented yet.')
    model.to(cuda_device)

    # Building the dataloader for training and validation (random 15% of the training)
    logging.info('Building dataset...')
    dataloader, val_dataloader = utils.create_dataloader(edge_list,
                                                         propagations,
                                                         polarities,
                                                         oversampling_minority,
                                                         training_type,
                                                         cuda_device,
                                                         batch_size
                                                         )

    logging.info('Dataset built.')

    # Setting Loss and Optimizer
    criterion = EchoChamberLoss(ec_prior_size, social_prior_size, propagation_prior_size, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop of the model
    model.train()
    loss_trend, nll_trend, kld_trend = [], [], []
    tot_loss_trend, tot_nll_trend, tot_kld_trend = [], [], []
    val_loss_trend, val_nll_trend, val_kld_trend = [], [], []
    tot_val_loss_trend, tot_val_nll_trend, tot_val_kld_trend = [], [], []
    estimation_methods = ['theta', 'phi', 'mixed', 'mixed_alt']
    val_stance_prediction_trend = {
        'phi': [], 'theta': [], 'mixed': [], 'mixed_alt': []
    }
    val_ec_trend = {
        'share': [], 'jaccard': [], 'mae': []
    }
    BEST_VAL_LOSS = np.inf
    BEST_VAL_METRIC = -np.inf
    annealer = utils.init_annealer(annealing, len(dataloader) * epochs)

    # Loop over epochs
    for epoch in range(epochs):
        logging.info(f'\n====\nEpoch: {epoch}/{epochs}')
        # Training
        logging.info(f'Annealing param: {round(criterion.anneal_param, 4)}')
        model, epoch_loss, epoch_nll, epoch_kld = utils.train_one_epoch(dataloader, optimizer,
                                                                        ablation, model,
                                                                        criterion, annealer, reweighting)
        loss_trend.append(np.mean(epoch_loss))
        tot_loss_trend += epoch_loss
        nll_trend.append(np.mean(epoch_nll))
        tot_nll_trend += epoch_nll
        kld_trend.append(np.mean(epoch_kld))
        tot_kld_trend += epoch_kld

        str_to_print = f'Training set Loss: {round(loss_trend[-1], 4)} === ' \
                       f'NLL: {round(nll_trend[-1], 4)} === ' \
                       f'KLD: {round(kld_trend[-1], 4)}'
        logging.info(str_to_print)

        # Validation
        val_epoch_loss, val_epoch_nll, val_epoch_kld = utils.eval_one_epoch(val_dataloader, model,
                                                                            criterion, ablation)
        val_loss_trend.append(np.mean(val_epoch_loss))
        tot_val_loss_trend += val_epoch_loss
        val_nll_trend.append(np.mean(val_epoch_nll))
        tot_val_nll_trend += val_epoch_nll
        val_kld_trend.append(np.mean(val_epoch_kld))
        tot_val_kld_trend += val_epoch_kld

        str_to_print = f'Validation set Loss: {round(val_loss_trend[-1], 4)} === ' \
                       f'NLL: {round(val_nll_trend[-1], 4)} === ' \
                       f'KLD: {round(val_kld_trend[-1], 4)}'
        logging.info(str_to_print)

        if val_epoch_loss[-1] < BEST_VAL_LOSS:
            torch.save(model.state_dict(), interim_data_path / 'best_val_model.pt')
            BEST_VAL_LOSS = val_epoch_loss[-1]

        # Validation on stance prediction
        val_stance_prediction_trend, BEST_VAL_METRIC, best_th = utils.eval_stance_prediction(link_adj, model,
                                                                                             cuda_device, data_dir,
                                                                                             propagations, polarities,
                                                                                             BEST_VAL_METRIC,
                                                                                             interim_data_path,
                                                                                             val_stance_prediction_trend,
                                                                                             logger, estimation_methods,
                                                                                             model_type, u2idx)

        # Validation on Echo-Chamber Analysis
        share_idx, jac_idx, mae_idx, comm_stats = utils.echo_chamber_analysis(model, link_adj,
                                                                              cuda_device, propagations,
                                                                              polarities, NPAIRS=20000,
                                                                              model_type=model_type)
        val_ec_trend['share'].append(share_idx)
        val_ec_trend['jaccard'].append(jac_idx)
        val_ec_trend['mae'].append(mae_idx)

        str_to_print = f'Validation EC share: {share_idx} === ' \
                       f'jaccard: {jac_idx} === ' \
                       f'MAE: {mae_idx}'
        logging.info(str_to_print)

        annealer.update()

    # Training is done! Save the final model
    torch.save(model.state_dict(), interim_data_path / 'model.pt')
    mlflow.log_artifact(interim_data_path / 'model.pt')
    mlflow.log_artifact(interim_data_path / 'best_val_model.pt')
    mlflow.log_artifact(interim_data_path / 'best_val_AUC_model.pt')

    # Plot section
    # Training loss
    fig = utils.plot_losses([loss_trend, nll_trend, kld_trend], 'Training loss')
    fig.savefig(interim_data_path / 'training_loss.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'training_loss.png')
    fig = utils.plot_losses([loss_trend, nll_trend, kld_trend], 'Training loss', log_scale=True)
    fig.savefig(interim_data_path / 'training_loss_logscale.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'training_loss_logscale.png')
    fig = utils.plot_losses([tot_loss_trend, tot_nll_trend, tot_kld_trend], 'Training loss', xlabel='Iteration')
    fig.savefig(interim_data_path / 'tot_training_loss.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'training_loss.png')
    fig = utils.plot_losses([tot_loss_trend, tot_nll_trend, tot_kld_trend], 'Training loss', log_scale=True,
                            xlabel='Iteration')
    fig.savefig(interim_data_path / 'tot_training_loss_logscale.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'tot_training_loss_logscale.png')

    # Validation loss
    fig = utils.plot_losses([val_loss_trend, val_nll_trend, val_kld_trend], 'Validation loss')
    fig.savefig(interim_data_path / 'validation_loss.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'validation_loss.png')
    fig = utils.plot_losses([val_loss_trend, val_nll_trend, val_kld_trend], 'Validation loss', log_scale=True)
    fig.savefig(interim_data_path / 'validation_loss_logscale.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'validation_loss_logscale.png')
    fig = utils.plot_losses([tot_val_loss_trend, tot_val_nll_trend, tot_val_kld_trend], 'Validation loss',
                            xlabel='Iteration')
    fig.savefig(interim_data_path / 'tot_validation_loss.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'tot_validation_loss.png')
    fig = utils.plot_losses([tot_val_loss_trend, tot_val_nll_trend, tot_val_kld_trend], 'Validation loss',
                            log_scale=True, xlabel='Iteration')
    fig.savefig(interim_data_path / 'tot_validation_loss_logscale.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'tot_validation_loss_logscale.png')

    # Validation AUC
    fig = utils.plot_val_auc(estimation_methods, val_stance_prediction_trend)
    fig.savefig(interim_data_path / f'val_AUC.png', dpi=200)
    mlflow.log_artifact(interim_data_path / f'val_AUC.png')

    for estimation_method in estimation_methods:
        mlflow.log_metric(f'AUC_{estimation_method}', val_stance_prediction_trend[estimation_method][-1])
        mlflow.log_metric(f'maxAUC_{estimation_method}', max(val_stance_prediction_trend[estimation_method]))

    # Echo-Chamber indexes
    comm_stats.to_csv(interim_data_path / 'comm_stats.csv')
    mlflow.log_artifact(interim_data_path / 'comm_stats.csv')
    comm_stats.to_html(interim_data_path / 'comm_stats.html')
    mlflow.log_artifact(interim_data_path / 'comm_stats.html')
    for ec_index in val_ec_trend:
        mlflow.log_metric(ec_index, val_ec_trend[ec_index][-1])

    fig = utils.plot_ec_indexes(val_ec_trend)
    fig.savefig(interim_data_path / f'val_EC_indexes.png', dpi=200)
    mlflow.log_artifact(interim_data_path / f'val_EC_indexes.png')

    # Purity Conductance
    fig = utils.plot_purity_conductance(model, link_adj, prop_adj, cuda_device, num_communities, best_th, model_type)
    fig.savefig(interim_data_path / f'purity_conductance.png', dpi=200)
    mlflow.log_artifact(interim_data_path / f'purity_conductance.png')

    mlflow.log_artifact(interim_data_path / 'log.txt')
    # Once all the files are logged into mlflow we can delete the interim directory
    return interim_data_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs='*', default=[12121995, ])
    parser.add_argument('--dataset', type=str, default="brexit", dest="dataset")
    parser.add_argument('--K', type=int, nargs='*', default=[8, ])
    parser.add_argument('--s', type=int, nargs='*', default=[8, ])
    parser.add_argument('--h', type=int, nargs='*', default=[16, ])
    parser.add_argument('--B', type=int, nargs='*', default=[5, ])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--oversampling', type=bool, default=True)
    parser.add_argument('--reweighting', type=bool, default=False)
    parser.add_argument('--annealing', type=str, default='constant')
    parser.add_argument('--link_removal', type=int, default=0)
    parser.add_argument('--prop_removal', type=bool, default=False)
    parser.add_argument('--prop_removal_perc', type=float, default=0.)
    parser.add_argument('--stance_detection_exp', type=bool, default=False)
    parser.add_argument('--ablation', type=str, default=models.DEFAULT_ABLATION)
    parser.add_argument('--training_type', type=str, default=models.DEFAULT_TRAINING_TYPE)
    parser.add_argument('--model_type', type=str, default='exact-posterior')
    parser.add_argument('--device', type=str, default='2')
    args = parser.parse_args()
    dataset_name = args.dataset
    mlflow.set_experiment(f'ECD_{dataset_name}')

    curr_dir = pathlib.Path.cwd()
    data_dir = curr_dir.parent / 'data' / 'raw' / dataset_name

    G = nx.read_edgelist(data_dir / 'edgelist.txt', create_using=nx.DiGraph, nodetype=int)
    excluded_edges = []
    if args.link_removal > 0:
        edges = list(G.edges)
        edges = np.random.permutation(edges)
        excluded_edges = edges[:args.link_removal]
        for edge in excluded_edges:
            G.remove_edge(edge[0], edge[1])

    link_adj = nx.to_numpy_array(G)
    edge_list = np.array(list(map(lambda x: list(x), list(G.edges()))))

    with open(data_dir / 'propagations_and_polarities.pkl', 'rb') as f:
        propagations, polarities = pickle.load(f)
        u2idx = None
        if dataset_name == "vaxNoVax":
            ITEM_SCORE_THRESHOLD = 0.5
            accepted = (np.abs(polarities) > ITEM_SCORE_THRESHOLD)
            propagations = [propagations[i] for i in np.where(accepted)[0]]
            polarities = polarities[accepted]
            # with open(data_dir / f'username2index.pkl', 'rb') as f2:
            #     u2idx = pickle.load(f2)
        # else:
        #     u2idx = None

    num_props_removed = 0
    removal_perc = args.prop_removal_perc  # 0.75
    retain_perc = 1. - removal_perc
    if args.prop_removal:
        new_propagations = []
        for propagation in propagations:
            prop_length = len(propagation)
            retained_length = math.ceil(retain_perc * prop_length)
            new_propagations.append(propagation[:retained_length if retained_length > 1 else prop_length])
            num_props_removed += prop_length-retained_length
        propagations = new_propagations

    if args.stance_detection_exp:
        # we remove all propagations of users for which we want to perform stance detection
        manual_stances = utils.open_manual_stances(path=data_dir / 'manual_stances.tsv',
                                                   include_username=u2idx is not None)
        users_to_exclude = set(manual_stances.user.values.tolist())
        new_propagations = []
        for propagation in propagations:
            cleaned_prop = list(set(propagation) - users_to_exclude)
            num_props_removed += (len(cleaned_prop) - len(propagation))
            new_propagations.append(cleaned_prop)
        propagations = new_propagations

    polarities = np.array(polarities)
    prop_adj = utils.generate_prop_adj(propagations, num_users=link_adj.shape[0])

    for seed in args.seed:
        for K in args.K:
            for batch_size in (4096,):
                for social_prior_size in args.s:
                    for ec_prior_size in args.h:
                        for propagation_prior_size in args.B:
                            with mlflow.start_run():
                                interim_data_path = run_experiment(
                                    dataset_name=dataset_name,
                                    link_adj=link_adj,
                                    prop_adj=prop_adj,
                                    edge_list=edge_list,
                                    polarities=polarities,
                                    propagations=propagations,
                                    u2idx=u2idx,
                                    num_communities=K,
                                    ec_prior_size=ec_prior_size,
                                    social_prior_size=social_prior_size,
                                    propagation_prior_size=propagation_prior_size,
                                    removed_links=excluded_edges,
                                    num_props_removed=num_props_removed,
                                    stance_detection_exp=args.stance_detection_exp,
                                    ablation=args.ablation,
                                    training_type=args.training_type,
                                    model_type=args.model_type,
                                    # Learning params
                                    epochs=args.epochs,
                                    batch_size=batch_size,
                                    learning_rate=args.lr,
                                    annealing=args.annealing,
                                    oversampling_minority=args.oversampling,
                                    reweighting=args.reweighting,
                                    seed=seed,
                                    device=args.device
                                )

                            try:
                                shutil.rmtree(interim_data_path, ignore_errors=True)
                            except OSError as e:
                                print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    main()
