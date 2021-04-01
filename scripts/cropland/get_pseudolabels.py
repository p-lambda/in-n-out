import argparse
import json
from pathlib import Path
from copy import deepcopy
import torch
from innout.main import setup, main, predict_classification
from innout.datasets import RangeDataset
from innout.load_utils import init_dataset, init_dataloader
import pickle
import numpy as np


def get_pseudolabels(model_dir_g, pseudolabel_dataset_config):
    # get the config from the model dir
    config_path = Path(model_dir_g) / 'config.json'
    with open(config_path, 'r') as f:
        config_g = json.load(f)
    config_g['restart_epoch_count'] = False
    setup(config_g, model_dir_g, no_wandb=True)
    model_g, _, _, _ = main(config_g, return_best=True)
    model_g.eval()

    pseudolabel_dataset_config['dataset']['args']['unlabeled_targets_path'] = None
    train_dataset = init_dataset(pseudolabel_dataset_config, 'train')

    # take the unlabeled part out
    unlabeled_start_idx = len(train_dataset.labeled_indices)
    unlabeled_end_idx = len(train_dataset)
    unlabeled_ratio = (unlabeled_end_idx - unlabeled_start_idx) / len(train_dataset)
    train_dataset = RangeDataset(train_dataset, unlabeled_start_idx, unlabeled_end_idx)
    train_loader = init_dataloader(pseudolabel_dataset_config, train_dataset, 'train', shuffle=False)
    device = torch.device('cuda')
    output = predict_classification(pseudolabel_dataset_config, model_g,
                                    device=device, loader=train_loader,
                                    is_binary=True)
    pred = np.squeeze(output > 0)
    # validate the correctness of pseudolabels
    targets = train_dataset.dataset._unseen_unlabeled_targets
    pseudolabel_acc = (pred == targets).mean()
    print(f'Pseudolabel acc: {pseudolabel_acc}')
    print(f'Number of pseudolabels: {len(pred)}')

    pseudolabel_dict = {'pseudolabels': pred,
                        'unlabeled_ratio': unlabeled_ratio,
                        'pseudolabel_acc': pseudolabel_acc}
    return pseudolabel_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get pseudolabels for self training')
    parser.add_argument('--model_dir', type=str, help='Directory of model for generating pseudolabels')
    parser.add_argument('--pseudolabel_path', type=str, help='path for saving pseudolabels')
    args = parser.parse_args()

    config_path = Path(args.model_dir) / 'config.json'
    with open(config_path, 'r') as f:
        config_g = json.load(f)
    pseudolabel_dataset_config = deepcopy(config_g)
    pseudolabel_dataset_config['dataset']['args']['use_unlabeled_id'] = True
    pseudolabel_dataset_config['dataset']['args']['use_unlabeled_ood'] = False

    pseudolabels_dict = get_pseudolabels(args.model_dir, pseudolabel_dataset_config)
    # save the pseudolabels
    with open(args.pseudolabel_path, 'wb') as pkl_file:
        pickle.dump(pseudolabels_dict, pkl_file, protocol=4)
