from losses import EchoChamberLoss
from gen_model import GenerativeModel

import os
import shutil
import itertools
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
from scipy.stats import pearsonr
from scipy.special import softmax

sns.set()


def run_experiment(
        link_adj,
        prop_adj,
        edge_list,
        polarities,
        propagations,
        num_communities,
        items_per_node,
        ec_prior_size=models.EC_PRIOR_SIZE,
        social_prior_size=models.SOCIAL_PRIOR_SIZE,
        propagation_prior_size=models.PROPAGATION_PRIOR_SIZE,
        removed_links=[],
        num_props_removed=0,
        ablation=models.DEFAULT_ABLATION,
        training_type=models.DEFAULT_TRAINING_TYPE,
        model_type='exact-posterior',
        # Learning params
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        oversampling_minority=True,
        reweighting=False,
        annealing='cyclic',
        seed=12121995,
        device="",
        targets={}
):
    mlflow.log_param('seed', seed)
    mlflow.log_param('ablation', ablation)
    mlflow.log_param('training_type', training_type)
    mlflow.log_param('link_removal', len(removed_links))
    mlflow.log_param('prop_removal', num_props_removed)
    mlflow.log_param('model_type', model_type)
    mlflow.log_param('num_communities', num_communities)
    mlflow.log_param('items_per_node', items_per_node)
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
    interim_data_path = curr_dir.parent / 'data' / 'interim' / f"{my_run_id}"
    if not os.path.exists(interim_data_path):
        os.makedirs(interim_data_path)

    # Set seed for reproducibility
    utils.set_seed(seed)

    # Set cuda visible and set cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = device
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
    print('Building dataset...')
    dataloader, val_dataloader = utils.create_dataloader(edge_list,
                                                         propagations,
                                                         polarities,
                                                         oversampling_minority,
                                                         training_type,
                                                         cuda_device,
                                                         batch_size
                                                         )

    logging.info('Dataset built.')
    print('Dataset built.')

    # Setting Loss and Optimizer
    criterion = EchoChamberLoss(ec_prior_size, social_prior_size, propagation_prior_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop of the model
    model.train()
    loss_trend, nll_trend, kld_trend = [], [], []
    tot_loss_trend, tot_nll_trend, tot_kld_trend = [], [], []
    val_loss_trend, val_nll_trend, val_kld_trend = [], [], []
    tot_val_loss_trend, tot_val_nll_trend, tot_val_kld_trend = [], [], []
    θ_mae_trend, φ_mae_trend, η_mae_trend = [], [], []
    polarities_pearson_trend = []
    val_ec_trend = {
        'share': [], 'jaccard': [], 'mae': []
    }
    BEST_VAL_LOSS = np.inf
    annealer = utils.init_annealer(annealing, len(dataloader) * epochs)

    # Loop over epochs
    for epoch in range(epochs):
        logging.info(f'\n====\nEpoch: {epoch}/{epochs}')
        print(f'\n====\nEpoch: {epoch}/{epochs}')
        # Training
        logging.info(f'Annealing param: {round(criterion.anneal_param, 4)}')
        print(f'Annealing param: {round(criterion.anneal_param, 4)}')
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
        print(str_to_print)

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
        print(str_to_print)

        if val_epoch_loss[-1] < BEST_VAL_LOSS:
            torch.save(model.state_dict(), interim_data_path / 'best_val_model.pt')
            BEST_VAL_LOSS = val_epoch_loss[-1]

        # Validation on estimating generating parameters
        with torch.no_grad():
            θ = model.theta(model.node_features, model.edge_index).detach().cpu().numpy()
            φ = softmax(model.phi(model.node_features, model.edge_index).detach().cpu().numpy(), axis=1)
            η = model.eta().detach().cpu().numpy()
            my_node_polarities = (θ*η).sum(1)
            real_node_polarities = (targets['θ'] * targets['η']).sum(1)
            r2 = pearsonr(my_node_polarities, real_node_polarities)[0]
            polarities_pearson_trend.append(r2)
            print(f'Pearson Polarities: {round(r2, 4)}')
            logging.info(f'Pearson Polarities: {round(r2, 4)}')
            # this step is done because we want to be sure that we are not failing because of order
            ordered_idxs = np.argsort(η)
            ord_η = η[ordered_idxs]
        # eta error
        target_mae = np.abs(ord_η - targets['η']).mean()
        print(f'MAE on η: {round(target_mae, 4)}')
        η_mae_trend.append(target_mae)
        # eval theta error
        best_val_mae = np.inf
        all_perm = list(itertools.permutations(range(θ.shape[1])))
        for perm in all_perm:
            ord_θ = θ[:, perm]
            target_mae = np.abs(ord_θ - targets['θ']).mean()
            if target_mae < best_val_mae:
                best_val_mae = target_mae
        logging.info(f'MAE on θ: {round(best_val_mae, 4)}')
        print(f'MAE on θ: {round(best_val_mae, 4)}')
        θ_mae_trend.append(best_val_mae)
        # eval phi error
        best_val_mae = np.inf
        all_perm = list(itertools.permutations(range(φ.shape[1])))
        for perm in all_perm:
            ord_φ = φ[:, perm]
            target_mae = np.abs(ord_φ - targets['φ']).mean()
            if target_mae < best_val_mae:
                best_val_mae = target_mae
        logging.info(f'MAE on φ: {round(best_val_mae, 4)}')
        print(f'MAE on φ: {round(best_val_mae, 4)}')
        φ_mae_trend.append(best_val_mae)

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
        print(str_to_print)

        annealer.update()

    # Training is done! Save the final model
    torch.save(model.state_dict(), interim_data_path / 'model.pt')
    
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

    # Target reconstruction error
    fig = utils.plot_target_reco([θ_mae_trend, φ_mae_trend, η_mae_trend])
    fig.savefig(interim_data_path / 'parameters_mae_trend.png', dpi=200)
    mlflow.log_artifact(interim_data_path / 'parameters_mae_trend.png')
    mlflow.log_metric('θ_mae', θ_mae_trend[-1])
    mlflow.log_metric('φ_mae', φ_mae_trend[-1])
    mlflow.log_metric('η_mae', η_mae_trend[-1])
    mlflow.log_metric('pearson_polarities', polarities_pearson_trend[-1])

    mlflow.log_metric('θ_best_mae', min(θ_mae_trend))
    mlflow.log_metric('φ_best_mae', min(φ_mae_trend))
    mlflow.log_metric('η_best_mae', min(η_mae_trend))
    mlflow.log_metric('polarities_best_pearson', max(polarities_pearson_trend))

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

    θ = model.theta(model.node_features, model.edge_index).detach().cpu().numpy()
    φ = softmax(model.phi(model.node_features, model.edge_index).detach().cpu().numpy(), axis=1)
    η = model.eta().detach().cpu().numpy()

    np.save(interim_data_path / 'θ_est_vs_real.npy', np.vstack([np.expand_dims(θ, 0), np.expand_dims(targets['θ'], 0)]))
    mlflow.log_artifact(interim_data_path / 'θ_est_vs_real.npy')
    np.save(interim_data_path / 'φ_est_vs_real.npy', np.vstack([np.expand_dims(φ, 0), np.expand_dims(targets['φ'], 0)]))
    mlflow.log_artifact(interim_data_path / 'φ_est_vs_real.npy')
    np.save(interim_data_path / 'η_est_vs_real.npy', np.vstack([np.expand_dims(η, 0), np.expand_dims(targets['η'], 0)]))
    mlflow.log_artifact(interim_data_path / 'η_est_vs_real.npy')

    mlflow.log_artifact(interim_data_path / 'log.txt')
    # Once all the files are logged into mlflow we can delete the interim directory
    return interim_data_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs='*', default=[325, 350, 375, 400, 450, 500, 525, 550, 575, 601])
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--eta', type=list, default=[-1., -0.5, 0., 0.5, 1.])
    parser.add_argument('--s', type=int, nargs='*', default=[8, ])
    parser.add_argument('--h', type=int, nargs='*', default=[16, ])
    parser.add_argument('--B', type=int, nargs='*', default=[5, ])
    parser.add_argument('--items_per_node', type=int, nargs='*', default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--oversampling', type=bool, default=True)
    parser.add_argument('--reweighting', type=bool, default=False)
    parser.add_argument('--annealing', type=str, default='constant')
    parser.add_argument('--ablation', type=str, default=models.DEFAULT_ABLATION)
    parser.add_argument('--training_type', type=str, default=models.DEFAULT_TRAINING_TYPE)
    parser.add_argument('--model_type', type=str, default='exact-posterior')
    parser.add_argument('--device', type=str, default='3')
    args = parser.parse_args()
    mlflow.set_experiment(f'ECD_synth')

    curr_dir = pathlib.Path.cwd()
    batch_size = 4096

    for seed in args.seed:
        for s in args.s:
            for h in args.h:
                for B in args.B:
                    for items_per_node in args.items_per_node:
                        utils.set_seed(seed)
                        synth_model = GenerativeModel(args.N, len(args.eta), M=(8 * args.N),
                                                      eta=args.eta,
                                                      social_prior_size=s,
                                                      ec_prior_size=h,
                                                      prop_prior_size=B,
                                                      social_beta=0.01,
                                                      ec_beta=1.)

                        zipf_exponent = 3.5
                        discard_above = 'N'
                        discard_below = 2
                        synth_model.generate_propagations(items_per_node * args.N,
                                                          zipf_exponent=zipf_exponent,
                                                          discard_above=discard_above,
                                                          discard_below=discard_below,
                                                          )

                        targets = {
                            'η': synth_model.η, 'θ': synth_model.θ, 'φ': synth_model.φ,
                            'node_polarities': (synth_model.θ*synth_model.η).sum(1)
                        }

                        G = synth_model.G

                        link_adj = nx.to_numpy_array(G)
                        edge_list = np.array(list(map(lambda x: list(x), list(G.edges()))))

                        propagations = synth_model.item2prop
                        polarities = synth_model.polarities

                        prop_adj = utils.generate_prop_adj(propagations, num_users=link_adj.shape[0])

                        with mlflow.start_run():
                            interim_data_path = run_experiment(
                                link_adj=link_adj,
                                prop_adj=prop_adj,
                                edge_list=edge_list,
                                polarities=polarities,
                                propagations=propagations,
                                num_communities=len(args.eta),
                                ec_prior_size=h,
                                social_prior_size=s,
                                propagation_prior_size=B,
                                removed_links=[],
                                num_props_removed=0,
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
                                device=args.device,
                                targets=targets,
                                items_per_node=items_per_node
                            )

                        try:
                            shutil.rmtree(interim_data_path, ignore_errors=True)
                        except OSError as e:
                            print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    main()
