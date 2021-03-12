import subprocess
import shlex
import argparse
import json
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
import datetime
import pandas as pd

from extrapolation.main import setup, main, predict_classification
from extrapolation import EXTRAPOLATION_ROOT
from extrapolation.datasets import RangeDataset
from extrapolation.load_utils import init_dataset, init_dataloader
from extrapolation.datasets import RangeDataset
from extrapolation.configs.celeba.gender_in_no_hats_out_hats.two_stage_attr_p2 import get_input_results, get_outputs_attributes_labels


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
    python_cmd = '/sailhome/ananya/extrapolation/.env/bin/python ' +\
           '/sailhome/ananya/extrapolation/extrapolation/main.py '
    python_cmd += opts
    return python_cmd


def run_sbatch(python_cmd, job_name='innout', output='logs', mail_type='END,FAIL', mail_user='eix@cs.stanford.edu',
               partition='jag-standard', exclude='jagupard[4-8,10-11]', nodes=1,
               gres='gpu:1', cpus_per_task=1, mem='16G'):

    cmd = f'sbatch --job-name={job_name} --output={output} --mail-type={mail_type} --mail-user={mail_user} ' +\
          f'--partition={partition} --exclude={exclude} --nodes={nodes} ' +\
          f'--gres={gres} --cpus-per-task={cpus_per_task} --mem={mem} /juice/scr/ananya/run_sbatch.sh '

    cmd += f'"{python_cmd}"'
    print(cmd)
    subprocess.run(shlex.split(cmd))


def run_job_chain(kwargs_list, job_name='ananya', output='logs', mail_type='END,FAIL',
                  mail_user='ananya@cs.stanford.edu', partition='jag-standard',
                  exclude='jagupard[4-8,10-11]', nodes=1, gres='gpu:1', cpus_per_task=1,
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


def run_baseline_exp(exp_name, model_dir, config_path, seed=0, lr=0.1):
    kwargs = {}
    kwargs = deepcopy(kwargs)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_baseline',
                   'seed': seed, 'project_name': 'extrapolation',
                   'dataset.args.seed': seed})
    kwargs_list = []
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs_list.append(kwargs.copy())
    run_job_chain(kwargs_list=kwargs_list, job_name=exp_name, output=f'/juice/scr/ananya/logs/{exp_name}')


def run_pretrain_exp(exp_name, model_dir, config_path, freeze_shared=False, seed=0, lr=0.03):
    kwargs = {}
    kwargs = deepcopy(kwargs)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_output',
                   'seed': seed, 'project_name': 'extrapolation',
                   'dataset.args.seed': seed})

    kwargs_list = []
    kwargs['model_dir'] = str(model_dir) + '_pretrain'
    kwargs['run_name'] = exp_name + '_pretrain'
    kwargs['dataset.train_args.split'] = 'all_unlabeled'
    kwargs_list.append(kwargs.copy())

    # resume from previous checkpoint
    kwargs['checkpoint_path'] = str(Path(kwargs['model_dir']) / 'best-checkpoint.pt')
    kwargs['dataset.train_args.split'] = 'train'
    # TODO changed to rerun
    kwargs['model_dir'] = Path(model_dir).parent / ('rerun_' + exp_name)
    kwargs['run_name'] = 'rerun_' + exp_name
    kwargs['loss.classname'] = 'torch.nn.BCEWithLogitsLoss'
    kwargs['eval_loss.classname'] = 'torch.nn.BCEWithLogitsLoss'
    kwargs['model.args.use_idx'] = 0
    kwargs['restart_epoch_count'] = True
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs['dataset.args.meta_as_target'] = False
    kwargs_list.append(kwargs.copy())
    run_job_chain(kwargs_list=kwargs_list, job_name=exp_name, output=f'logs/%j_{exp_name}')


def run_input_p1(exp_name, model_dir, config_path, seed=0, lr=0.1):
    kwargs = {}
    kwargs = deepcopy(kwargs)
    print(exp_name)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_inputs',
                   'seed': seed, 'project_name': 'extrapolation',
                   'dataset.args.seed': seed})
    print(kwargs)
    kwargs_list = []
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs_list.append(kwargs.copy())
    run_job_chain(kwargs_list=kwargs_list, job_name=exp_name, output=f'/juice/scr/ananya/logs/{exp_name}')
    # cmd = get_python_cmd(kwargs)
    # cmd += ' && '
    # cmd += ('/sailhome/ananya/extrapolation/.env/bin/python '
    #         '/sailhome/ananya/extrapolation/extrapolation/configs/'
    #         'celeba/gender_in_no_hats_out_hats/two_stage_attr_p2.py '
    #         '')
    # get python command for first job
    # get python command for p2
    # chain them, call sbatch


def run_baseline_experiments(lr=0.1, num_trials=5):
    for i in range(num_trials):
        run_baseline_exp(
            exp_name='baseline_celeba_' + str(lr) + '_' + str(i),
            model_dir='/juice/scr/ananya/extrapolation_experiments/celeba_baseline_' + str(lr) + '_' + str(i),
            config_path='/sailhome/ananya/extrapolation/extrapolation/configs/celeba/gender_in_no_hats_out_hats/baseline_resnet.yaml',
            seed=i,
            lr=lr)


def run_pretrain_experiments(lr=0.03, num_trials=5):
    for i in range(num_trials):
        run_pretrain_exp(
            exp_name='output_celeba_' + str(lr) + '_' + str(i),
            model_dir='/juice/scr/ananya/extrapolation_experiments/celeba_output_' + str(lr) + '_' + str(i),
            config_path='/sailhome/ananya/extrapolation/extrapolation/configs/celeba/gender_in_no_hats_out_hats/output_model_p1.yaml',
            seed=i,
            lr=lr)


def run_input_experiments_p1(lr=0.1, num_trials=5):
    for i in range(num_trials):
        exp_name = 'input_celeba_' + str(lr) + '_' + str(i)
        model_dir = '/juice/scr/ananya/extrapolation_experiments/celeba_input_' + str(lr) + '_' + str(i)
        config_path='/sailhome/ananya/extrapolation/extrapolation/configs/celeba/gender_in_no_hats_out_hats/two_stage_attr_p1.yaml'
        run_input_p1(exp_name, model_dir, config_path, seed=i, lr=lr)


def run_input_experiments_p2(lr=0.1, num_trials=5):
    scores = []
    for i in range(num_trials):
        exp_id = f'celeba_input_{lr}_{i}'
        model_dir = f'/juice/scr/ananya/extrapolation_experiments/{exp_id}'
        pseudolabel_save_path = f'/sailhome/ananya/extrapolation/pseudolabels/{exp_id}'
        scores.append(get_input_results(model_dir, pseudolabel_save_path=pseudolabel_save_path))
    print(scores)
    print('mean: ', np.mean(scores, axis=0))
    print('stddev: ', np.std(scores, axis=0))


def run_selftraining_exp(exp_name, config_path_f, kwargs_f=None,
                         input_mode=True, trial=0, lr=0.1, unlabeled_weight=0.5, pretrain=True):

    input_exp_id = f'celeba_input_0.1_{trial}'
    # model_dir = f'/juice/scr/ananya/extrapolation_experiments/{exp_id}'
    pseudolabel_save_path = Path(f'/sailhome/ananya/extrapolation/pseudolabels/{input_exp_id}.npy')

    output_checkpoint = Path(f'/juice/scr/ananya/extrapolation_experiments/celeba_output_0.1_{trial}_pretrain') / 'best-checkpoint.pt'

    model_dir_root = Path('/u/scr/ananya/extrapolation_experiments/')
    model_dir_f = model_dir_root / exp_name
    if kwargs_f is None:
        kwargs_f = {}

    kwargs_f = deepcopy(kwargs_f)
    kwargs_f.update({'config': config_path_f, 'model_dir': model_dir_f,
                   'run_name': exp_name, 'group_name': 'celeba_selftrain',
                   'seed': trial, 'project_name': 'extrapolation',
                   'dataset.args.seed': trial,
                   'loss.args.unlabeled_weight': unlabeled_weight,
                   'dataset.train_args.pseudolabel_path': pseudolabel_save_path})
    if pretrain:
        kwargs_f['checkpoint_path'] = output_checkpoint
    # resume from previous checkpoint
    kwargs_f['run_name'] = exp_name
    kwargs_f['model.args.use_idx'] = 0
    kwargs_f['restart_epoch_count'] = True
    kwargs_f['optimizer.args.lr'] = lr
    kwargs_f['scheduler.args.lr'] = kwargs_f['optimizer.args.lr']
    kwargs_f['dataset.args.meta_as_target'] = False

    python_cmd = get_python_cmd(kwargs_f)
    run_sbatch(python_cmd, job_name=exp_name, output=f'logs/%j_{exp_name}')


def run_selftraining_exps(input_mode=True, lr=0.1, unlabeled_weight=0.5, pretrain=True):
    if input_mode:
        exp_type = 'celeba_innout'
        config_path_f = EXTRAPOLATION_ROOT / 'configs/celeba/gender_in_no_hats_out_hats/innout.yaml'
    else:
        raise ValueError("not implemented")
        # exp_type = 'output_selftrain'
        # config_path_f = EXTRAPOLATION_ROOT / 'configs/landcover_v2/CNN1DSelfTrainOutput.yaml'

    if not pretrain:
        exp_type += '_no_pretrain'

    for trial in range(5):
        kwargs = {'overwrite': False,
                  'no_wandb': False,
                  'return_best': True,
                  'restart_epoch_count': True}

        run_selftraining_exp(
                f'{exp_type}_ulw{unlabeled_weight}_{lr}_trial{trial}',
                config_path_f=config_path_f,
                kwargs_f=kwargs,
                input_mode=input_mode,
                trial=trial, lr=lr,
                unlabeled_weight=unlabeled_weight,
                pretrain=pretrain)


if __name__ == "__main__":
    # run_pretrain_experiments()
    # run_pretrain_experiments(lr=0.1)
    run_baseline_experiments()
    # run_baseline_experiments(lr=0.03)
    # run_input_experiments_p1()
    # run_input_experiments_p2()
    # run_selftraining_exps(unlabeled_weight=0.5)
    # for ulw in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    #     run_selftraining_exps(unlabeled_weight=ulw)
    # for ulw in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        # run_selftraining_exps(unlabeled_weight=ulw, pretrain=False)

