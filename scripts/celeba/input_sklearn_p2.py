from tqdm import tqdm
import argparse
import innout.datasets.celeba as celeba
from torch.utils.data import DataLoader
from importlib import reload  
import pandas
from functools import partial
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from PIL import Image
import pickle
import innout.models.resnet
import innout.main as main
from innout.load_utils import init_dataset
from sklearn.linear_model import LogisticRegression


def get_outputs_attributes_labels(data_loader, model):
    model_outputs = []
    attributes = []
    labels = []
    feats = []
    for i_batch, batch in enumerate(data_loader):
        if 'target' in batch:
            labels.append(batch['target'].numpy())
        attributes.append(batch['domain_label']['meta'].numpy())
        cur_model_outputs, cur_features = model.forward(batch['data'], with_feats=True)
        feats.append(cur_features.cpu().detach().numpy())
        cur_model_outputs = cur_model_outputs.cpu().detach().numpy()
        model_outputs.append(cur_model_outputs)
    model_outputs = np.concatenate(model_outputs)
    attributes = np.concatenate(attributes)
    if len(labels) > 0:
        labels = np.concatenate(labels)
    features = np.concatenate(feats)
    return model_outputs, features, attributes, labels


def accuracy(preds, attributes, labels, attribute_selector):
    mask = [attribute_selector(a) for a in attributes]
    indices = np.where(mask)
    # print(np.sum(mask))
    return np.mean(preds[indices] == labels[indices]), np.mean(labels[indices])

model_dir_cache = {}
dataset_cache = {}

def get_input_results(model_dir, pseudolabels_dir=None, attrs=None):
    model_dir = Path(model_dir).resolve().expanduser()
    with open(model_dir / 'config.json', 'r') as f:
        l = f.read()
        config = json.loads(l)
    config['dataset']['args']['in_labeled_split_idx'] = 1
    config['dataset']['args']['in_labeled_splits'] = [1800, 200]

    if str(model_dir) not in model_dir_cache:
        main.setup(config, model_dir, no_wandb=True, setup_log=False)
        model, train_df, eval_df, ret_epoch = main.main(config)
    else:
        model = None

    def make_dataset(split):
        if (model_dir, split) in dataset_cache:
            return dataset_cache[(model_dir, split)]
        dataset = init_dataset(config, dataset_type=split)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        dataset_cache[(model_dir, split)] = (dataset, dataloader)
        return dataset, dataloader

    def get_xs_ys(dataloader, model, dataset, split):
        if str(model_dir) in model_dir_cache and split in model_dir_cache[str(model_dir)]:
            model_outputs, features, attributes, labels = model_dir_cache[str(model_dir)][split]
        else:
            model_outputs, features, attributes, labels = get_outputs_attributes_labels(dataloader, model)
            if str(model_dir) not in model_dir_cache:
                model_dir_cache[str(model_dir)] = {}
            model_dir_cache[str(model_dir)][split] = (model_outputs, features, attributes, labels)

        if attrs is not None:
            # add different attrs
            attr_idxs = [celeba.attr_name_to_idx[attr_name] for attr_name in attrs]
            attributes = np.asarray([dataset._attr[i][attr_idxs].numpy() for i in dataset._indices])

        xs = np.concatenate([model_outputs, attributes], axis=1)
        ys = np.squeeze(labels)
        return xs, ys

    def get_acc(clf, xs, ys):
        return np.mean(clf.predict(xs) == ys)

    train_dataset, train_dataloader = make_dataset('train')
    val_dataset, val_dataloader = make_dataset('val')
    test_dataset, test_dataloader = make_dataset('test')
    test2_dataset, test2_dataloader = make_dataset('test2')
    in_unlabeled_dataset, in_unlabeled_dataloader = make_dataset('in_unlabeled')

    # Try training with a few regularization values. See what does well on in-domain set.
    train_xs, train_ys = get_xs_ys(train_dataloader, model, train_dataset, 'train')
    val_xs, val_ys = get_xs_ys(val_dataloader, model, val_dataset, 'val')
    test_xs, test_ys = get_xs_ys(test_dataloader, model, test_dataset, 'test')
    test2_xs, test2_ys = get_xs_ys(test2_dataloader, model, test2_dataset, 'test2')
    in_unlabeled_xs, _ = get_xs_ys(in_unlabeled_dataloader, model, in_unlabeled_dataset, 'in_unlabeled')

    models = []
    accs = []
    Cs = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
    for C in tqdm(Cs):
        clf = LogisticRegression(C=C)
        clf.fit(train_xs, train_ys)
        train_acc = get_acc(clf, train_xs, train_ys)
        val_acc = get_acc(clf, val_xs, val_ys)
        # print(val_acc)
        test_acc = get_acc(clf, test_xs, test_ys)
        test2_acc = get_acc(clf, test2_xs, test2_ys)
        # Warning: careful about flipping the order or adding more accs:
        # we use this ordering later on when constructing a scores
        # dictionary, so need to edit that too.
        accs.append((train_acc, val_acc, test_acc, test2_acc))
        models.append(clf)

    accs = np.asarray(accs)
    best_val_idx = np.argmax(accs[:, 1])
    scores = accs[best_val_idx, :]
    # print('Train acc: {}'.format(scores[0] * 100))
    # print('Val acc: {}'.format(scores[1] * 100))
    # print('Test acc: {}'.format(scores[2] * 100))
    # print('Test2 acc: {}'.format(scores[3] * 100))

    # make in unlabeled predictions
    if pseudolabels_dir is not None:
        pseudolabel_save_path = pseudolabels_dir + '/pseudolabels'
        best_model = models[best_val_idx]
        with open(str(pseudolabel_save_path) + "_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        in_unlabeled_preds = best_model.predict(in_unlabeled_xs)
        # in_unlabeled_targets = np.asarray([in_unlabeled_dataset._attr[index][in_unlabeled_dataset._target_attribute].squeeze().numpy() for index in in_unlabeled_dataset._indices])
        # pseudolabel_acc = np.mean(in_unlabeled_preds == in_unlabeled_targets)
        np.save(pseudolabel_save_path, in_unlabeled_preds)

    # Save scores as json in model_dir.
    scores_dict = {
        'train_acc': scores[0],
        'val_acc': scores[1],
        'test_acc': scores[2],
        'test2_acc': scores[3],
    }
    results_file = model_dir / 'aux_in_results.json'
    with open(results_file, "w") as outfile: 
            json.dump(scores_dict, outfile)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run second phase of input model.')
    parser.add_argument('--model_dir', type=str, metavar='c',
                        help='Model directory path.', required=True)
    parser.add_argument('--pseudolabels_dir', type=str, metavar='c',
                        help='Directory (should exist) to save pseudolabels.', required=False)
    args, unparsed = parser.parse_known_args()

    # model_dir = '/sailhome/ananya/extrapolation/extrapolation/models/two_stage_attr_p1'
    get_input_results(args.model_dir, args.pseudolabels_dir)
    
