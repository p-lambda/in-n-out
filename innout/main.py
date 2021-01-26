import argparse
import logging
from pathlib import Path
import itertools
from collections import defaultdict

import yaml
import json
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score

from innout.load_utils import update_config, initialize, get_params
from innout.load_utils import init_dataset, init_dataloader
from innout.utils import now_to_str, DataParallel, to_device


def save_ckp(epoch, model, optimizer, scheduler, model_dir, chkpt_name):
    checkpoint_fpath = str(model_dir / chkpt_name)
    logging.info(f"Saving to checkpoint {checkpoint_fpath}")
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, checkpoint_fpath)


def load_ckp(checkpoint_fpath, model, optimizer=None, scheduler=None, reset_optimizer=False):
    logging.info(f"Loading from checkpoint {checkpoint_fpath}")
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    epoch = 0
    if not reset_optimizer:
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            epoch = int(checkpoint_fpath.split('epoch')[1].split('.')[0])
    return epoch


def train(config, model, device, train_loader, optimizer, scheduler, epoch):
    if 'updater' in config:
        update_epoch = initialize(config['updater'])
        update_epoch(config, model, device, train_loader, optimizer, scheduler, epoch)

    train_loss = initialize(config['loss'], {'model': model, 'reduction': 'mean'})
    model.train()
    train_metrics = []
    for batch_idx, batch in enumerate(train_loader):
        # TODO does this handle batch norm?
        # model.zero_grad()
        if hasattr(train_loss, 'custom_input') and train_loss.custom_input:
            # uses domain label?
            batch_metrics = train_loss(batch, epoch, optimizer, device)
        else:
            data, target, metadata = batch['data'], batch['target'], batch['domain_label']
            data, target = to_device(data, device), to_device(target, device)
            if hasattr(train_loss, 'requires_domain') and train_loss.requires_domain:
                loss = train_loss(model(data), target, metadata)
            else:
                loss = train_loss(model(data), target)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            batch_metrics = {'epoch': epoch, 'loss': loss.item()}
        if 'epoch' not in batch_metrics or 'loss' not in batch_metrics:
            raise ValueError('epoch and loss must be in batch_metrics')
        loss = batch_metrics['loss']
        train_metrics.append(batch_metrics)

        # print progress
        if batch_idx % config['log_interval'] == 0:
            num_batches = 1 if config['batch_gd'] else len(train_loader)
            logging.info(
                f'Train Epoch: {epoch} [{batch_idx*config["batch_size"]}/{config["epoch_size"]} ({100.*batch_idx / num_batches :.2f}%)]\tLoss: {loss :.6f}')
            if len(batch_metrics) > 2:
                logging.info(f"METRICS: {batch_metrics}")

        # if batch mode we need to break (hack)
        if config['batch_gd']:
            break
    scheduler.step()
    return train_metrics


def eval(config, model, device, split, loader, visualizer=None, return_metric_totals=False):
    metric_totals = defaultdict(float)
    # TODO: IRM loss does not work for eval
    if 'eval_loss' in config:
        eval_loss = initialize(config['eval_loss'], {'eval': True, 'model': model, 'split': split})
    else:
        eval_loss = nn.CrossEntropyLoss(reduction='sum')

    # XXX a hack for handling binary classification
    is_binary = isinstance(eval_loss, nn.BCEWithLogitsLoss)

    total_batches = 1 if config['batch_gd'] else len(loader)

    if split == 'train':
        num_batches = config.get('eval_train_batches')
        if num_batches is not None:
            total_batches = min(num_batches, total_batches)

    # plot_batch_idx = np.random.randint(total_batches)
    is_custom = hasattr(eval_loss, 'custom_input') and eval_loss.custom_input
    plot_batch_idx = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if is_custom:
                # uses domain label?
                batch_metrics = eval_loss(batch, None, None, device)
                for k in batch_metrics.keys():
                    if k == 'epoch':
                        continue
                    else:
                        metric_totals[k] += batch_metrics[k]
                # we expect metric_totals to contain correct, total
                pred = None
            else:
                data, target, metadata = batch['data'], batch['target'], batch['domain_label']
                data = to_device(data, device)
                target = to_device(target, device)

                # -100 is default value
                mask = (target != -100)
                if len(mask.shape) >= 2 and mask.shape[1] == 1:
                    mask = mask.squeeze(-1)

                if not isinstance(data, list) and not isinstance(data, tuple) and len(mask.shape) == 1:
                    if isinstance(data, dict):  # TODO: this feels hacky
                        data['data'] = data['data'][mask]
                    else:
                        data = data[mask]
                    target = target[mask]
                output = model(data)
                if hasattr(eval_loss, 'requires_domain') and eval_loss.requires_domain:
                    metric_totals['loss'] += eval_loss(output, target, metadata).item()
                else:
                    metric_totals['loss'] += eval_loss(output, target).item()

                if is_binary:
                    pred = (output > 0.).long()
                    target = target.long()
                else:
                    pred = output.max(1, keepdim=True)[1]
                metric_totals['total'] += mask.sum().item()
                # for AUROC, AUPRC
                if 'probs' not in metric_totals or 'targets' not in metric_totals:
                    metric_totals['probs'] = []
                    metric_totals['targets'] = []

                if is_binary:
                    positive_probs = torch.sigmoid(output)
                    negative_probs = 1 - positive_probs
                    probs = torch.cat([negative_probs, positive_probs], dim=1)
                else:
                    probs = torch.nn.functional.softmax(output, dim=1)
                metric_totals['probs'].append(probs.detach().cpu().numpy())
                metric_totals['targets'].append(target.detach().cpu().numpy())

            if batch_idx == plot_batch_idx and visualizer is not None:
                visualizer.update(batch, pred=pred)

            if split == 'train' and (batch_idx + 1) >= total_batches:
                break
            # if batch mode we need to break (hack)
            if config['batch_gd']:
                break
    if is_custom:
        total = metric_totals['total']
        for k in metric_totals.keys():
            if k in {'epoch', 'total'}:
                continue
            metric_totals[k] /= total
        loss = metric_totals['loss']
        logging.info(
            f'{split.upper()}: ' +
            ', '.join([f'{k}: {v :.4f}' for k, v in metric_totals.items()]))
        eval_data = {f'{split}_{k}': v for k, v in metric_totals.items()}
    else:
        total = metric_totals['total']
        loss = metric_totals['loss'] / total
        # AUROC, AUPRC
        target = np.concatenate(metric_totals['targets'], axis=0)
        probs = np.concatenate(metric_totals['probs'], axis=0)
        if is_binary and len(target.shape) >= 2:
            target = target.squeeze(axis=-1)
        num_classes = probs.shape[1]
        target_onehot = np.eye(num_classes)[target]
        auroc = roc_auc_score(target_onehot, probs)
        auprc = average_precision_score(target_onehot, probs)

        pred = np.argmax(probs, axis=1)
        correct_mask = (target == pred)
        correct = correct_mask.sum()
        accuracy = correct_mask.mean()

        eval_data = {'loss': loss, 'accuracy': accuracy, 'auroc': auroc, 'auprc': auprc}
        for label in range(num_classes):
            label_mask = (target == label)
            class_accuracy = (pred[label_mask] == target[label_mask]).mean()
            eval_data[f'accuracy_class_{label}'] = class_accuracy

        eval_data = {f"{split}_{k}": v for k, v in eval_data.items()}
        logging.info(
            f'{split.upper()}: Loss: {loss :.4f}, '
            f'Accuracy: {correct}/{total} ({100.0 * accuracy :.2f}%) '
            f'AUROC: {auroc :.3f}, AUPRC: {auprc :.3f}')

    # do visualization
    if visualizer is not None:
        visualizer.visualize(split)

    if not return_metric_totals:
        return eval_data
    else:
        return eval_data, metric_totals


def predict_classification(config, model, device, loader, is_binary=False):
    '''
    returns classification outputs from the model
    '''
    metric_totals = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data, target, metadata = batch['data'], batch['target'], batch['domain_label']
            data = to_device(data, device)
            target = to_device(target, device)
            output = model(data)
            metric_totals['output'].append(output.detach().cpu().numpy())
            # if batch mode we need to break (hack)
            if config['batch_gd']:
                break
    output = np.concatenate(metric_totals['output'], axis=0)
    return output


def batch_loader(dataset, shuffle=True):
    data, targets, domains = dataset.data, dataset.targets, dataset.domain_labels

    if shuffle:
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)
        data, targets, domains = data[idxs], targets[idxs], domains[idxs]
    return itertools.cycle([{'data': data, 'target': targets, 'domain_label': domains}])


def main(config, overwrite=False, checkpoint_path=None, return_best=True, restart_epoch_count=False):
    no_wandb = config['no_wandb']
    done_training = False
    if not overwrite:
        final_checkpoint_path = model_dir / f"checkpoint-epoch{config['epochs']}.pt"
        if final_checkpoint_path.exists():
            logging.info(f'Done training - final checkpoint found at {final_checkpoint_path}! Loading the saved model from best checkpoint')
            checkpoint_path = model_dir / "best-checkpoint.pt"
            done_training = True

    if not done_training:

        train_dataset = init_dataset(config, 'train')
        eval_train_dataset = init_dataset(config, 'eval_train', train_dataset)
        val_dataset = init_dataset(config, 'val', train_dataset)
        test_dataset = init_dataset(config, 'test', train_dataset)
        test2_dataset = init_dataset(config, 'test2', train_dataset)
        config['epoch_size'] = len(train_dataset)

        if config['batch_gd']:
            train_loader = batch_loader(train_dataset, shuffle=True)
            eval_train_loader = batch_loader(eval_train_dataset, shuffle=True)
            val_loader = batch_loader(val_dataset, shuffle=True)
            test_loader = batch_loader(test_dataset, shuffle=True)
            test2_loader = batch_loader(test2_dataset, shuffle=True)
        else:
            train_loader = init_dataloader(config, train_dataset, 'train')
            eval_train_loader = init_dataloader(config, eval_train_dataset, 'eval_train')
            val_loader = init_dataloader(config, val_dataset, 'val')
            test_loader = init_dataloader(config, test_dataset, 'test')
            test2_loader = init_dataloader(config, test2_dataset, 'test2')

        model = initialize(config['model'])
        if config['use_cuda']:
            model = DataParallel(model).cuda()

        optimizer = initialize(config['optimizer'], {'params': get_params(model)})
        scheduler = initialize(config['scheduler'], {'optimizer': optimizer})
        if 'visualizer' in config:
            visualizer = initialize(config['visualizer'], {'no_wandb': no_wandb, 'model': model})
        else:
            visualizer = None

        if not checkpoint_path:
            start_epoch = 0
            # save initial checkpoint
            save_ckp(start_epoch, model, optimizer, scheduler, model_dir, 'initial.pt')
        else:
            start_epoch = load_ckp(checkpoint_path, model, optimizer, scheduler, reset_optimizer=restart_epoch_count)

        train_metrics = []
        eval_metrics = []
        for epoch in range(start_epoch + 1, config['epochs'] + 1):
            train_data = train(
                    config, model, device, train_loader, optimizer, scheduler, epoch)
            train_metrics += train_data

            logging.info(120 * '=')
            eval_data = {'epoch': int(epoch)}
            eval_data.update(eval(config, model, device, 'train', eval_train_loader, visualizer))
            eval_data.update(eval(config, model, device, 'val', val_loader, visualizer))
            eval_data.update(eval(config, model, device, 'test', test_loader, visualizer))
            if 'test2_args' in config['dataset']:
                eval_data.update(eval(config, model, device, 'test2', test2_loader, visualizer))
            logging.info(120 * '=')
            eval_metrics.append(eval_data)

            if not no_wandb:
                wandb.log(eval_data)

            # save stats
            train_df = pd.DataFrame(train_metrics)
            eval_df = pd.DataFrame(eval_metrics)

            # save checkpoint
            if epoch % config['save_freq'] == 0 or epoch == config['epochs']:
                save_ckp(epoch, model, optimizer, scheduler, model_dir,
                         f'checkpoint-epoch{epoch}.pt')
                if epoch == config['epochs']:
                    save_ckp(epoch, model, optimizer, scheduler, model_dir,
                             'last-checkpoint.pt')

            # if best, also save
            early_stop_metric = config.get('early_stop_metric', 'val_accuracy')
            early_stop_max = config.get('early_stop_max_best', True)
            if early_stop_metric in eval_df:
                if early_stop_max and eval_df[early_stop_metric].idxmax() == (len(eval_df) - 1):
                    save_ckp(epoch, model, optimizer, scheduler, model_dir,
                             'best-checkpoint.pt')
                if not early_stop_max and eval_df[early_stop_metric].idxmin() == (len(eval_df) - 1):
                    save_ckp(epoch, model, optimizer, scheduler, model_dir,
                             'best-checkpoint.pt')
            else:
                logging.warn(f"{early_stop_metric} was not found as a metric. Using val_loss")
                # if no acc, use loss
                if eval_df['val_loss'].idxmin() == (len(eval_df) - 1):
                    save_ckp(epoch, model, optimizer, scheduler, model_dir,
                             'best-checkpoint.pt')

            if epoch == config['epochs']:
                train_df.to_csv(str(model_dir / 'stats_train.tsv'), sep='\t')
                eval_df.to_csv(str(model_dir / 'stats_eval.tsv'), sep='\t')
    else:
        optimizer = None
        scheduler = None
        model = initialize(config['model'])
        if config['use_cuda']:
            model = DataParallel(model).cuda()

    # return the best model or last model
    train_df = pd.read_csv(str(model_dir / 'stats_train.tsv'), sep='\t')
    eval_df = pd.read_csv(str(model_dir / 'stats_eval.tsv'), sep='\t')

    if return_best:
        best_checkpoint_path = model_dir / "best-checkpoint.pt"
        best_epoch = load_ckp(best_checkpoint_path, model, optimizer, scheduler)
        logging.info(f"Evaluation stats at best epoch {best_epoch}:")
        logging.info(eval_df[eval_df['epoch'] == best_epoch].iloc[0])
        ret_epoch = best_epoch
    else:
        last_epoch = config['epochs']
        last_checkpoint_path = model_dir / f"checkpoint-epoch{last_epoch}.pt"
        _ = load_ckp(last_checkpoint_path, model, optimizer, scheduler)
        logging.info(f"Evaluation stats at last epoch {last_epoch}:")
        logging.info(eval_df[eval_df['epoch'] == last_epoch].iloc[0])
        ret_epoch = last_epoch
    return model, train_df, eval_df, ret_epoch


def setup(config, save_model_dir, seed=None, no_wandb=False, run_name=None, project_name=None, group_name=None, entity_name=None, setup_log=True):
    global model_dir
    model_dir = Path(save_model_dir).resolve().expanduser()
    model_dir.mkdir(exist_ok=True, parents=True)

    if setup_log:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(str(model_dir / 'training.log')),
                logging.StreamHandler()
            ])
        # logger = logging.getLogger()

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed + 111)
    # should provide some improved performance
    cudnn.benchmark = True
    global device
    device = torch.device("cuda" if config['use_cuda'] else "cpu")
    global dl_kwargs
    if config['use_cuda']:
        dl_kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        dl_kwargs = {}

    config['no_wandb'] = no_wandb
    if not no_wandb:
        run_name = now_to_str() if run_name is None else run_name
        run_obj = wandb.init(project=project_name, name=run_name, group=group_name, entity=entity_name, reinit=True)
        config['wandb_url'] = run_obj.get_url()
        config['run_name'] = run_name
        config['group_name'] = group_name
        config['entity_name'] = entity_name
        wandb.config.update(config)

    config_json = model_dir / 'config.json'
    if not config_json.exists():
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(config, f)
        if setup_log:
            logging.info(str(config))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run model')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--model_dir',
                        help='directory of model for saving checkpoint', required=True)
    parser.add_argument('--seed', type=int, metavar='s', default=None,
                        help='random seed')
    parser.add_argument('--no_wandb', action='store_true', help='disable W&B')
    parser.add_argument('--project_name', default=None, help='Name of the wandb project')
    parser.add_argument('--group_name', default=None, help='Name of the wandb group (a group of runs)')
    parser.add_argument('--run_name', default=None, help='Name of the wandb run')
    parser.add_argument('--entity_name', default='p-lambda', help='Name of the team')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing model')
    parser.add_argument('--return_best', action='store_true', help='At the end, display results of best model')
    parser.add_argument('--checkpoint_path', default=None, help='Resume checkpoint from this path')
    parser.add_argument('--restart_epoch_count', action='store_true', help='Start training from epoch 0')

    args, unparsed = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # update config with args
    update_config(unparsed, config)
    config['checkpoint_path'] = args.checkpoint_path
    config['restart_epoch_count'] = args.restart_epoch_count
    config['model_dir'] = args.model_dir
    config['config'] = args.config
    config['seed'] = args.seed

    setup(config,
          args.model_dir,
          seed=args.seed,
          no_wandb=args.no_wandb,
          run_name=args.run_name,
          project_name=args.project_name,
          group_name=args.group_name,
          entity_name=args.entity_name)

    main(config, overwrite=args.overwrite,
         checkpoint_path=args.checkpoint_path,
         return_best=args.return_best,
         restart_epoch_count=args.restart_epoch_count)
