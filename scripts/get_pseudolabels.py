import argparse
import json
from pathlib import Path
from copy import deepcopy
import torch
from innout.main import setup, main, predict_classification
from innout.load_utils import init_dataset, init_dataloader
import numpy as np


def get_pseudolabels(args, pseudolabel_dataset_config):
    # get the config from the model dir
    model_dir_g = args.model_dir
    config_path = Path(model_dir_g) / 'config.json'
    with open(config_path, 'r') as f:
        config_g = json.load(f)
    config_g['restart_epoch_count'] = False
    setup(config_g, model_dir_g, no_wandb=True)
    model_g, _, _, _ = main(config_g, return_best=True)
    model_g.eval()

    pseudolabel_dataset_config['dataset']['args']['unlabeled_targets_path'] = None
    train_dataset = init_dataset(pseudolabel_dataset_config, 'train')
    train_dataset = train_dataset.get_unlabeled_dataset()
    train_loader = init_dataloader(pseudolabel_dataset_config, train_dataset,
                                   'train', shuffle=False)
    device = torch.device('cuda')

    output = predict_classification(pseudolabel_dataset_config, model_g,
                                    device=device, loader=train_loader,
                                    is_binary=args.is_binary)

    if args.is_binary:
        pred = np.squeeze(output > 0)
    else:
        pred = output.argmax(1)

    print(f'Number of pseudolabels: {len(pred)}')
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get pseudolabels for self training')
    parser.add_argument('--model_dir', type=str, help='Directory of model for generating pseudolabels')
    parser.add_argument('--pseudolabel_path', type=str, help='path for saving pseudolabels')
    parser.add_argument('--use_unlabeled_id', action='store_true', help='use unlabeled id')
    parser.add_argument('--use_unlabeled_ood', action='store_true', help='use_unlabeled ood')
    parser.add_argument('--is_binary', action='store_true', help='binary classification')
    args = parser.parse_args()

    config_path = Path(args.model_dir) / 'config.json'
    with open(config_path, 'r') as f:
        config_g = json.load(f)
    pseudolabel_dataset_config = deepcopy(config_g)
    pseudolabel_dataset_config['dataset']['args']['use_unlabeled_id'] = args.use_unlabeled_id
    pseudolabel_dataset_config['dataset']['args']['use_unlabeled_ood'] = args.use_unlabeled_ood

    pseudolabels = get_pseudolabels(args, pseudolabel_dataset_config)
    # save the pseudolabels
    np.save(args.pseudolabel_path, pseudolabels)
