import subprocess
import shlex
import argparse
import json
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
import datetime
import shutil

from innout import INNOUT_ROOT
from innout.main import setup, main, predict_classification
from innout.datasets import RangeDataset
from innout.load_utils import init_dataset, init_dataloader


def get_time_str():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    return dt_string


def get_python_cmd(kwargs=None):
    if kwargs is not None:
        opts = ''.join([f"--{k}={v} " for k, v in kwargs.items() if not isinstance(v, bool) or '.' in k])
        opts += ''.join([f"--{k} " for k, v in kwargs.items() if isinstance(v, bool) and v and '.' not in k])
    else:
        opts = ''
    python_cmd = '/sailhome/rmjones/in-n-out/.env/bin/python ' +\
        '/sailhome/rmjones/in-n-out/innout/main.py '
    python_cmd += opts
    return python_cmd


def run_sbatch(python_cmd, job_name='rmjones', output='logs',
               mail_type='END,FAIL', mail_user='rmjones@cs.stanford.edu',
               partition='jag-standard', exclude='jagupard[4-8,10,28-29]',
               nodes=1, gres='gpu:1', cpus_per_task=1, mem='16G'):

    cmd = f'sbatch --job-name={job_name} --output={output} --mail-type={mail_type} --mail-user={mail_user} ' +\
          f'--partition={partition} --exclude={exclude} --nodes={nodes} ' +\
          f'--gres={gres} --cpus-per-task={cpus_per_task} --mem={mem} /u/scr/eix/run_sbatch.sh '

    cmd += f'"{python_cmd}"'
    print(cmd)
    if not args.dryrun:
        subprocess.run(shlex.split(cmd))


def run_job_chain(kwargs_list, job_name='rmjones', output='logs', mail_type='END,FAIL',
                  mail_user='rmjones@cs.stanford.edu', partition='jag-standard',
                  exclude='jagupard[4-8,10,28-29]', nodes=1, gres='gpu:1', cpus_per_task=1,
                  mem='16G'):
    '''
    kwargs_list: list of dict
        list of kwargs
    '''
    cmd = ''
    for kwargs in kwargs_list:
        python_cmd = get_python_cmd(kwargs)
        if len(cmd) > 0:
            cmd += ' && '
        cmd += python_cmd
    run_sbatch(cmd, job_name=job_name, output=output, mail_type=mail_type,
               mail_user=mail_user, partition=partition, exclude=exclude,
               nodes=nodes, gres=gres, cpus_per_task=cpus_per_task, mem=mem)


def run_exp(exp_name, config_path, kwargs):
    model_dir = model_dir_root / f'landcover_{exp_name}'
    if kwargs is None:
        kwargs = {}
    kwargs.update({'config': config_path, 'model_dir': model_dir, 'run_name': exp_name})

    python_cmd = get_python_cmd(kwargs)
    run_sbatch(python_cmd, job_name=exp_name, output=f'logs/%j_{exp_name}')


def run_base_exps():
    exp_type = 'cnn1d'

    config_path = INNOUT_ROOT / 'configs/landcover/CNN1D.yaml'

#    for trial in [1, 2, 3,4,5]:
    for trial in [1]:
        kwargs = {'dataset.args.unlabeled_prop': args.unlabeled_prop,
                  'epochs': 400,
                  'scheduler.num_epochs': 400,
                  'overwrite': args.overwrite,
                  'no_wandb': args.no_wandb,
                  'return_best': args.return_best,
                  'seed': trial+111,
                  'dataset.args.seed': trial,
                  'group_name': 'LandcoverSanityCheck'}

        run_exp(f'era5asfeature_{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{trial}', config_path, kwargs)
        # don't add as feature
        kwargs.update({'dataset.args.include_ERA5': False, 'model.args.in_channels': 8})
        run_exp(f'noera5_{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{trial}', config_path, kwargs)


def run_pretrain_exp(exp_name, config_path, kwargs, use_unlabeled_id=True, use_unlabeled_ood=True, seed=1111):
    model_dir = model_dir_root / f'landcover_{exp_name}'
    if kwargs is None:
        kwargs = {}

    kwargs = deepcopy(kwargs)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'LandcoverSanityCheck',
                   'seed': seed+111,
                   'dataset.args.seed': seed})

    kwargs_list = []
    kwargs['dataset.args.pretrain'] = True
    kwargs['dataset.args.use_unlabeled_id'] = use_unlabeled_id
    kwargs['dataset.args.use_unlabeled_ood'] = use_unlabeled_ood
    kwargs['epochs'] = 200
    kwargs['scheduler.args.num_epochs'] = kwargs['epochs']
    kwargs['model.args.use_idx'] = 1
    kwargs['model_dir'] = str(model_dir) + '_pretrain'
    kwargs['run_name'] = exp_name + '_pretrain'
    kwargs['optimizer.args.lr'] = 0.1
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs_list.append(kwargs.copy())

    # resume from previous checkpoint
    if args.use_last_pretrain:
        kwargs['checkpoint_path'] = str(Path(kwargs['model_dir']) / 'last-checkpoint.pt')
    else:
        kwargs['checkpoint_path'] = str(Path(kwargs['model_dir']) / 'best-checkpoint.pt')
    kwargs['model_dir'] = model_dir
    kwargs['run_name'] = exp_name
    kwargs['dataset.args.pretrain'] = False
    kwargs['dataset.args.use_unlabeled_id'] = False
    kwargs['dataset.args.use_unlabeled_ood'] = False
    kwargs['loss.classname'] = 'torch.nn.CrossEntropyLoss'
    kwargs['eval_loss.classname'] = 'torch.nn.CrossEntropyLoss'
    kwargs['model.args.use_idx'] = 0
    kwargs['restart_epoch_count'] = True
    kwargs['epochs'] = 400
    kwargs['optimizer.args.lr'] = 0.01
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs_list.append(kwargs.copy())
    run_job_chain(kwargs_list=kwargs_list, job_name=exp_name, output=f'logs/%j_{exp_name}')


def run_pretrain_exps():
    exp_type = 'era5asout_pretrain'

    config_path = INNOUT_ROOT / 'configs/landcover/CNN1DPretrain.yaml'

    # add as feature
    kwargs = {'overwrite': args.overwrite,
              'no_wandb': args.no_wandb,
              'return_best': args.return_best,
              'dataset.args.unlabeled_prop': args.unlabeled_prop}

    if args.standardize_unlabeled_sample_size:
        kwargs['dataset.args.standardize_unlabeled_sample_size'] = True

    # for trial in [1, 2]:
    #for trial in [1,2,3,4,5]:
    for trial in [1]:
        run_pretrain_exp(f'{exp_type}_unlabeledprop{args.unlabeled_prop}_nounlabeled_trial{trial}',
                         config_path, kwargs, use_unlabeled_id=False, use_unlabeled_ood=False, seed=trial)
        run_pretrain_exp(f'{exp_type}_unlabeledprop{args.unlabeled_prop}_onlyunlabeledid_trial{trial}',
                         config_path, kwargs, use_unlabeled_id=True, use_unlabeled_ood=False, seed=trial)
        run_pretrain_exp(f'{exp_type}_unlabeledprop{args.unlabeled_prop}_onlyunlabeledood_trial{trial}',
                         config_path, kwargs, use_unlabeled_id=False, use_unlabeled_ood=True, seed=trial)
        run_pretrain_exp(f'{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{trial}',
                         config_path, kwargs, use_unlabeled_id=True, use_unlabeled_ood=True, seed=trial)


def get_pseudolabels(model_dir_g, pseudolabel_dataset_config, get_unlabeled_ratio=False):
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
    unlabeled_start_idx = len(train_dataset.data)
    unlabeled_end_idx = len(train_dataset)
    unlabeled_ratio = (unlabeled_end_idx - unlabeled_start_idx) / len(train_dataset)
    train_dataset = RangeDataset(train_dataset, unlabeled_start_idx, unlabeled_end_idx)
    train_loader = init_dataloader(pseudolabel_dataset_config, train_dataset, 'train', shuffle=False)
    device = torch.device("cuda")
    output = predict_classification(pseudolabel_dataset_config, model_g, device=device, loader=train_loader, is_binary=False)
    pred = output.argmax(1)

    # validate the correctness of pseudolabels
    targets = train_dataset.dataset._unseen_unlabeled_targets
    pseudolabel_acc = (pred == targets).mean()
    print(f"Pseudolabel acc: {pseudolabel_acc}")
    print(f"Number of pseudolabels: {len(pred)}")

    if get_unlabeled_ratio:
        return pred, unlabeled_ratio
    else:
        return pred


def run_selftraining_exp(exp_name, model_dir_g, config_path_f, kwargs_f=None,
                         seed=None, do_pseudolabels=True,
                         unlabeled_weight=0.5,
                         # unlabeled_weight_over_unlabeled_ratio=None,
                         input_mode=True, get_cmd_only=False, use_all_unlabeled=False):

    pseudolabel_dir = Path('/u/scr/rmjones/innout-pseudolabels')
    pseudolabel_dir.mkdir(exist_ok=True)
    pseudolabel_path = pseudolabel_dir / f'{exp_name}.npy'
    if do_pseudolabels:
        config_path = Path(model_dir_g) / 'config.json'
        with open(config_path, 'r') as f:
            config_g = json.load(f)
        pseudolabel_dataset_config = deepcopy(config_g)
        if input_mode:
            pseudolabel_dataset_config['dataset']['args']['use_unlabeled_id'] = True
            pseudolabel_dataset_config['dataset']['args']['use_unlabeled_ood'] = False
        else:
            pseudolabel_dataset_config['dataset']['args']['use_unlabeled_id'] = False
            pseudolabel_dataset_config['dataset']['args']['use_unlabeled_ood'] = True

        if use_all_unlabeled:
            pseudolabel_dataset_config['dataset']['args']['use_unlabeled_id'] = True
            pseudolabel_dataset_config['dataset']['args']['use_unlabeled_ood'] = True

        pseudolabel_dataset_config['dataset.args.seed'] = seed
        pseudolabels_id, unlabeled_ratio = get_pseudolabels(model_dir_g, pseudolabel_dataset_config, get_unlabeled_ratio=True)
        # save the pseudolabels
        np.save(str(pseudolabel_path), pseudolabels_id)

    # unlabeled_weight = unlabeled_weight_over_unlabeled_ratio * unlabeled_ratio
    exp_name += f'_unlabeledweight{unlabeled_weight}'

    model_dir_f = model_dir_root / f'landcover{exp_name}'
    if kwargs_f is None:
        kwargs_f = {}

    kwargs_f = deepcopy(kwargs_f)
    kwargs_f.update({'config': config_path_f, 'model_dir': model_dir_f,
                   'loss.args.unlabeled_weight': unlabeled_weight,
                   'run_name': exp_name, 'group_name': 'LandcoverSanityCheck',
                   'seed': seed+111, 'dataset.args.seed': seed,
                   'dataset.args.unlabeled_targets_path': pseudolabel_path})

    python_cmd = get_python_cmd(kwargs_f)
    if not get_cmd_only:
        run_sbatch(python_cmd, job_name=exp_name, output=f'logs/%j_{exp_name}')
    else:
        return python_cmd


def run_selftraining_exps(input_mode=True):
    if input_mode:
        exp_type = 'input_selftrain'
        config_path_f = INNOUT_ROOT / 'configs/landcover/CNN1DSelfTrainInput.yaml'
        if args.use_baseline_pseudolabels:
            exp_type = 'basepseudolabels_selftrain'
    else:
        exp_type = 'output_selftrain'
        config_path_f = INNOUT_ROOT / 'configs/landcover/CNN1DSelfTrainOutput.yaml'

    # for trial in [1, 2]:
    for trial in [1,2,3,4,5]:
        # was relevant when we tried many unlabeled_weight
        do_pseudolabels = True

        for unlabeled_weight in [0.1, 0.3, 0.5, 0.7, 0.9]:

            kwargs = {'overwrite': args.overwrite,
                      'no_wandb': args.no_wandb,
                      'return_best': args.return_best,
                      'dataset.args.unlabeled_prop': args.unlabeled_prop,
                      'restart_epoch_count': True}
            if args.z_noise_std > 0.0:
                kwargs['dataset.args.z_noise_std'] = args.z_noise_std
                kwargs['optimizer.args.lr'] = 0.1
                kwargs['scheduler.args.lr'] = 0.1
            if args.standardize_unlabeled_sample_size:
                kwargs['dataset.args.standardize_unlabeled_sample_size'] = True

            if args.large_finetune_lr:
                kwargs['optimizer.args.lr'] = 0.1
                kwargs['scheduler.args.lr'] = 0.1

            if input_mode:
                if args.use_baseline_pseudolabels:
                    model_dir_g = model_dir_root / f'landcover_noera5_cnn1d_unlabeledprop{args.unlabeled_prop}_trial{trial}'
                else:
                    model_dir_g = model_dir_root / f'landcover_era5asfeature_cnn1d_unlabeledprop{args.unlabeled_prop}_trial{trial}'
                if args.use_last_pretrain:
                    checkpoint_path_f = model_dir_root / f'landcover_era5asout_pretrain_unlabeledprop{args.unlabeled_prop}_trial{trial}_pretrain' / 'last-checkpoint.pt'
                else:
                    checkpoint_path_f = model_dir_root / f'landcover_era5asout_pretrain_unlabeledprop{args.unlabeled_prop}_trial{trial}_pretrain' / 'best-checkpoint.pt'
                kwargs.update({'checkpoint_path': checkpoint_path_f})

            else:
                model_dir_g = model_dir_root / f'landcover_era5asout_pretrain_unlabeledprop{args.unlabeled_prop}_trial{trial}'

            # calculate unlabeled_weight as the validation accuracy of the model_g * ratio of unlabeled examples in the finetune dataset

            # g_stats_eval = model_dir_g / 'stats_eval.tsv'
            # df = pd.read_csv(g_stats_eval, sep='\t')
            # best_idx = df['val_accuracy'].idxmax()
            # g_val_acc = df.iloc[best_idx].to_dict()['val_accuracy']
            # # still need to multiply by unlabeled_ratio
            # unlabeled_weight_over_unlabeled_ratio = g_val_acc

            run_selftraining_exp(
                    f'{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{trial}',
                    model_dir_g=model_dir_g, config_path_f=config_path_f,
                    kwargs_f=kwargs, seed=trial, do_pseudolabels=do_pseudolabels,
                    unlabeled_weight=unlabeled_weight,
                    # unlabeled_weight_over_unlabeled_ratio=unlabeled_weight_over_unlabeled_ratio,
                    input_mode=input_mode)
            do_pseudolabels = False


def run_selftraining_exps_iterated(input_mode=True, iterations=2):
    use_all_unlabeled = False
    if input_mode:
        exp_type = 'input_selftrain'
        config_path_f = INNOUT_ROOT / 'configs/landcover/CNN1DSelfTrainInput.yaml'
        if args.use_baseline_pseudolabels:
            exp_type = 'basepseudolabels_selftrain'
    else:
        use_all_unlabeled = True

    if use_all_unlabeled:
        exp_type = 'all_selftrain'
        config_path_f = INNOUT_ROOT / 'configs/landcover/CNN1DSelfTrainInput.yaml'

    for trial in [1,2,3,4,5]:
    # for trial in [5]:
        # was relevant when we tried many unlabeled_weight
        do_pseudolabels = True
        unlabeled_weight = 0.5

        kwargs = {'overwrite': args.overwrite,
                  'no_wandb': args.no_wandb,
                  'return_best': args.return_best,
                  'dataset.args.unlabeled_prop': args.unlabeled_prop,
                  'restart_epoch_count': True,
                  'epochs': 400}

        kwargs['optimizer.args.lr'] = 0.1
        kwargs['scheduler.args.lr'] = 0.1

        # XXX hack - trial 5 was going to nan
        if trial == 5:
            kwargs['optimizer.args.lr'] = 0.05
            kwargs['scheduler.args.lr'] = 0.05

        if use_all_unlabeled:
            kwargs['dataset.args.use_unlabeled_id'] = True
            kwargs['dataset.args.use_unlabeled_ood'] = True

        base_exp_id = f'{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{trial}'

        dropout = 0.5

        for st_iteration in range(iterations):
            if input_mode or use_all_unlabeled:
                if st_iteration == 0:
                    # model_dir_g = model_dir_root / f'landcover_era5asfeature_cnn1d_unlabeledprop{args.unlabeled_prop}_trial{trial}'
                    # checkpoint_path_f = model_dir_root / f'landcover_era5asout_pretrain_unlabeledprop{args.unlabeled_prop}_trial{trial}_pretrain' / 'best-checkpoint.pt'
                    # just copy over
                    base_exp_id_innout = f'input_selftrain_unlabeledprop{args.unlabeled_prop}_trial{trial}'
                    innout_src = model_dir_root / ('landcover_' + base_exp_id_innout + f'_unlabeledweight{unlabeled_weight}')
                    checkpoint_path_f_tgt = model_dir_root / ('landcover_' + base_exp_id + f'_iter0_dropout{dropout}_unlabeledweight{unlabeled_weight}')

                    if not Path(checkpoint_path_f_tgt).exists():
                        shutil.copytree(innout_src, checkpoint_path_f_tgt)
                    continue
                else:
                    prev_exp_id = 'landcover_' + base_exp_id + f'_iter{st_iteration - 1}_dropout{dropout}_unlabeledweight{unlabeled_weight}'
                    model_dir_g = model_dir_root / prev_exp_id
                    checkpoint_path_f = model_dir_g / 'best-checkpoint.pt'

                    # add some regularization
                    kwargs['optimizer.args.weight_decay'] = 0.0
                    kwargs['model.args.dropout_prob'] = dropout

                kwargs.update({'checkpoint_path': checkpoint_path_f})
            else:
                raise ValueError("input_mode must be True, not impl")

            curr_exp_id = base_exp_id + f'_iter{st_iteration}_dropout{dropout}'

            cmd = run_selftraining_exp(
                    curr_exp_id,
                    model_dir_g=model_dir_g, config_path_f=config_path_f,
                    kwargs_f=kwargs, seed=trial, do_pseudolabels=do_pseudolabels,
                    unlabeled_weight=unlabeled_weight,
                    input_mode=input_mode,
                    get_cmd_only=True,
                    use_all_unlabeled=use_all_unlabeled)
            subprocess.run(shlex.split(cmd))



def run_std_selftraining_exps(with_z=False):
    exp_type = 'std_selftrain'
    if with_z:
        exp_type = 'std_selftrain_withz'
    config_path_f = INNOUT_ROOT / 'configs/landcover/CNN1DSelfTrainInput.yaml'

    for trial in [1,2,3,4,5]:
        # was relevant when we tried many unlabeled_weight
        do_pseudolabels = True

        for unlabeled_weight in [0.1, 0.3, 0.5, 0.7, 0.9]:

            kwargs = {'overwrite': args.overwrite,
                      'no_wandb': args.no_wandb,
                      'return_best': args.return_best,
                      'dataset.args.unlabeled_prop': args.unlabeled_prop,
                      'restart_epoch_count': True,
                      'optimizer.args.lr': 0.1,
                      'scheduler.args.lr': 0.1,
                      }
            if args.z_noise_std > 0.0:
                kwargs['dataset.args.z_noise_std'] = args.z_noise_std

            if with_z:
                kwargs['model.args.in_channels'] = 14
                kwargs['dataset.args.include_ERA5'] = True

            model_dir_g = model_dir_root / f'landcover_era5asfeature_cnn1d_unlabeledprop{args.unlabeled_prop}_trial{trial}'

            # calculate unlabeled_weight as the validation accuracy of the model_g * ratio of unlabeled examples in the finetune dataset

            # g_stats_eval = model_dir_g / 'stats_eval.tsv'
            # df = pd.read_csv(g_stats_eval, sep='\t')
            # best_idx = df['val_accuracy'].idxmax()
            # g_val_acc = df.iloc[best_idx].to_dict()['val_accuracy']
            # # still need to multiply by unlabeled_ratio
            # unlabeled_weight_over_unlabeled_ratio = g_val_acc

            run_selftraining_exp(
                    f'{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{trial}',
                    model_dir_g=model_dir_g, config_path_f=config_path_f,
                    kwargs_f=kwargs, seed=trial, do_pseudolabels=do_pseudolabels,
                    unlabeled_weight=unlabeled_weight,
                    # unlabeled_weight_over_unlabeled_ratio=unlabeled_weight_over_unlabeled_ratio,
                    input_mode=True)
            do_pseudolabels = False

# def run_masked_pretrain_exp(exp_name, config_path, kwargs, use_unlabeled=True, seed=1111):
#     model_dir = model_dir_root / f'landcover_{exp_name}'
#     if kwargs is None:
#         kwargs = {}
# 
#     kwargs = deepcopy(kwargs)
#     kwargs.update({'config': config_path, 'model_dir': model_dir,
#                    'run_name': exp_name, 'group_name': 'landcover_pretrain',
#                    'seed': seed+111,
#                    'dataset.args.seed': seed})
# 
#     kwargs_list = []
#     kwargs['dataset.args.masked_pretrain'] = True
#     kwargs['dataset.args.use_unlabeled_id'] = use_unlabeled
#     kwargs['dataset.args.use_unlabeled_ood'] = use_unlabeled
#     kwargs['epochs'] = 200
#     kwargs['scheduler.args.num_epochs'] = kwargs['epochs']
#     kwargs['model_dir'] = str(model_dir) + '_pretrain'
#     kwargs['optimizer.args.lr'] = 0.1
#     kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
#     kwargs_list.append(kwargs.copy())
# 
#     # resume from previous checkpoint
#     kwargs['checkpoint_path'] = str(Path(kwargs['model_dir']) / 'best-checkpoint.pt')
#     kwargs['model_dir'] = model_dir
#     kwargs['dataset.args.masked_pretrain'] = False
#     kwargs['dataset.args.use_unlabeled_id'] = False
#     kwargs['dataset.args.use_unlabeled_ood'] = False
#     kwargs['loss.classname'] = 'torch.nn.CrossEntropyLoss'
#     kwargs['eval_loss.classname'] = 'torch.nn.CrossEntropyLoss'
#     kwargs['model.args.use_idx'] = 0
#     kwargs['restart_epoch_count'] = True
#     kwargs['epochs'] = 400
#     kwargs['optimizer.args.lr'] = 0.1
#     kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
#     kwargs_list.append(kwargs.copy())
#     run_job_chain(kwargs_list=kwargs_list, job_name=exp_name, output=f'logs/%j_{exp_name}')
# 
# 
# def run_masked_pretrain_exps():
#     exp_type = 'masked_pretrain'
#     config_path = INNOUT_ROOT / 'configs/landcover/CNN1DMaskedPretrain.yaml'
# 
#     # add as feature
#     kwargs = {'overwrite': args.overwrite,
#               'no_wandb': args.no_wandb,
#               'return_best': args.return_best}
# 
#     for trial in [1, 2]:
#         kwargs.update({'dataset.args.unlabeled_prop': 0.4})
#         run_masked_pretrain_exp(f'{exp_type}_unlabeledprop0.4_nounlabeled_trial{trial}', config_path, kwargs, use_unlabeled=False, seed=trial)
#         kwargs.update({'dataset.args.unlabeled_prop': 0.4})
#         run_masked_pretrain_exp(f'{exp_type}_unlabeledprop0.4_trial{trial}', config_path, kwargs, use_unlabeled=True, seed=trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts')
    parser.add_argument('--return_best', action='store_true', default=False,
                        help='do early stopping')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite')
    parser.add_argument('--no_wandb', action='store_true', default=False,
                        help='no wandb')
    parser.add_argument('--run_selftrain', action='store_true', default=False,
                        help='selftrain exps')
    parser.add_argument('--dryrun', action='store_true', default=False,
                        help='dryrun')
    parser.add_argument('--unlabeled_prop', type=float, default=0.9,
                        help='proportion of unlabeled data')
    parser.add_argument('--use_last_pretrain', action='store_true', default=False,
                        help='pretrain checkpoint to use')
    parser.add_argument('--use_baseline_pseudolabels', action='store_true', default=False,
                        help='use baseline model to pseudolabel ID unlabeled data')
    parser.add_argument('--standardize_unlabeled_sample_size', action='store_true', default=False,
                        help='make unlabeled data size equal across selftrain exps')

    args = parser.parse_args()
    model_subdir = f'landcover_unlabeledprop_{args.unlabeled_prop}_normalizeunlabeled'
    if args.standardize_unlabeled_sample_size:
        model_subdir += "_standardizeunlabeled"

    model_dir_root = INNOUT_ROOT.parent / 'models' / model_subdir
    model_dir_root.mkdir(exist_ok=True, parents=True)
    seed = 1111
    splits = ['train', 'val', 'test']
    root = INNOUT_ROOT.parent / 'data'

    if args.run_selftrain:
        # run_selftraining_exps(input_mode=True)

        run_selftraining_exps_iterated(input_mode=True, iterations=2)

        # run_selftraining_exps(input_mode=False)
        # run_std_selftraining_exps(with_z=False)
        # run_std_selftraining_exps(with_z=True)
    else:
#        run_base_exps()
        run_pretrain_exps()
