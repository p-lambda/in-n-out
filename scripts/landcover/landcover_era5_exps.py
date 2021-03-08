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
INNOUT_ROOT_PARENT = INNOUT_ROOT.parent


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
    if args.use_cl:
        python_cmd = f'python ' +\
            f'{INNOUT_ROOT}/main.py '
    else:
        python_cmd = f'{INNOUT_ROOT_PARENT}/.env/bin/python ' +\
            f'{INNOUT_ROOT}/main.py '
    python_cmd += opts
    return python_cmd


def run_sbatch(python_cmd, job_name='landcover', nodes=1,
               output='exp', gres='gpu:1', cpus_per_task=1, mem='16G', cl_extra_deps=None):

    if args.use_slurm:
        logpath = INNOUT_ROOT_PARENT / 'logs' / output
        logpath.parent.mkdir(exist_ok=True)
        run_sbatch_script = INNOUT_ROOT_PARENT / 'run_sbatch.sh'
        cmd = f'sbatch --job-name={job_name} --output={logpath} ' +\
              f'--partition={args.slurm_partition} --exclude={args.slurm_exclude} --nodes={nodes} ' +\
              f'--gres={gres} --cpus-per-task={cpus_per_task} --mem={mem} {run_sbatch_script} '

        cmd += f'"{python_cmd}"'
    elif args.use_cl:
        cl_extra_deps_str = ' '.join([f":{dep}" for dep in cl_extra_deps])
        cmd = f'cl run -n {job_name} -w in-n-out-iclr --request-docker-image ananya/in-n-out \
                --request-gpus 1 --request-memory 16g --request-queue tag=nlp \
                :innout :configs :landcover_data.pkl {cl_extra_deps_str} '
        if cl_extra_deps is None:
            cmd += f'"export PYTHONPATH=.; mkdir models; mkdir innout-pseudolabels; {python_cmd}"'
        else:
            cl_extra_deps_cps = ' '.join([f"cp -r {dep}/models/* models;" for dep in cl_extra_deps])
            cmd += f'"export PYTHONPATH=.; mkdir models; mkdir innout-pseudolabels; {cl_extra_deps_cps} {python_cmd}"'
    else:
        cmd = python_cmd
    print(cmd)
    if not args.dryrun:
        subprocess.run(shlex.split(cmd))

def run_job_chain(kwargs_list, job_name='landcover', output='logs',
                  nodes=1, gres='gpu:1', cpus_per_task=1, mem='16G'):
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

    run_sbatch(cmd, job_name=job_name, output=output,
               nodes=nodes, gres=gres, cpus_per_task=cpus_per_task, mem=mem)


def run_exp(exp_name, config_path, kwargs):
    model_dir = model_dir_root / f'landcover_{exp_name}'
    if kwargs is None:
        kwargs = {}
    kwargs.update({'config': config_path, 'model_dir': model_dir, 'run_name': exp_name})

    python_cmd = get_python_cmd(kwargs)
    run_sbatch(python_cmd, job_name=exp_name, output=exp_name)


def run_base_exps(aux_inputs=False):
    config_path = INNOUT_ROOT_PARENT / 'configs/landcover/CNN1D.yaml'

    kwargs = {'dataset.args.unlabeled_prop': args.unlabeled_prop,
              'epochs': 400,
              'scheduler.num_epochs': 400,
              'overwrite': args.overwrite,
              'no_wandb': args.no_wandb,
              'return_best': True,
              'seed': args.trial+111,
              'dataset.args.seed': args.trial,
              'group_name': 'landcover'}

    if aux_inputs:
        run_exp(f'aux-inputs_unlabeledprop{args.unlabeled_prop}_trial{args.trial}', config_path, kwargs)
    else:
        # don't add as feature
        kwargs.update({'dataset.args.include_ERA5': False, 'model.args.in_channels': 8})
        run_exp(f'baseline_unlabeledprop{args.unlabeled_prop}_trial{args.trial}', config_path, kwargs)


def run_pretrain_exp(exp_name, config_path, kwargs, use_unlabeled_id=True, use_unlabeled_ood=True, seed=1111):
    model_dir = model_dir_root / f'landcover_{exp_name}'
    if kwargs is None:
        kwargs = {}

    kwargs = deepcopy(kwargs)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'landcover',
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
    run_job_chain(kwargs_list=kwargs_list, job_name=exp_name, output=exp_name)


def run_pretrain_exps(use_unlabeled_id=True, use_unlabeled_ood=True):
    exp_type = 'aux-outputs'

    config_path = INNOUT_ROOT_PARENT / 'configs/landcover/CNN1DPretrain.yaml'

    # add as feature
    kwargs = {'overwrite': args.overwrite,
              'no_wandb': args.no_wandb,
              'return_best': True,
              'dataset.args.unlabeled_prop': args.unlabeled_prop}

    if args.standardize_unlabeled_sample_size:
        kwargs['dataset.args.standardize_unlabeled_sample_size'] = True

    exp_name = f'{exp_type}_unlabeledprop{args.unlabeled_prop}'
    if use_unlabeled_id and not use_unlabeled_ood:
        exp_name += '_only_unlabeled_id'
    if not use_unlabeled_id and use_unlabeled_ood:
        exp_name += '_only_unlabeled_ood'
    exp_name += f'_trial{args.trial}'
    run_pretrain_exp(exp_name,
                     config_path, kwargs,
                     use_unlabeled_id=use_unlabeled_id,
                     use_unlabeled_ood=use_unlabeled_ood,
                     seed=args.trial)


def run_selftraining_exp(exp_name, model_dir_g, config_path_f, kwargs_f=None,
                         seed=None, do_pseudolabels=True,
                         unlabeled_weight=0.5,
                         get_cmd_only=False):

    pseudolabel_dir = INNOUT_ROOT_PARENT / 'innout-pseudolabels'
    if not args.use_cl:
        pseudolabel_dir.mkdir(exist_ok=True, parents=True)
    pseudolabel_path = pseudolabel_dir / f'{exp_name}.npy'
    python_cmd = ""
    if do_pseudolabels:
        get_pseudolabels_script = INNOUT_ROOT_PARENT / 'scripts' / 'get_pseudolabels.py'
        python_cmd = f'{INNOUT_ROOT_PARENT}/.env/bin/python ' +\
                     f'{get_pseudolabels_script} --model_dir {model_dir_g} ' +\
                     f'--pseudolabel_path {pseudolabel_path} --seed {seed} && '

    exp_name += f'_unlabeledweight{unlabeled_weight}'

    model_dir_f = model_dir_root / f'landcover_{exp_name}'
    if kwargs_f is None:
        kwargs_f = {}

    kwargs_f = deepcopy(kwargs_f)
    kwargs_f.update({'config': config_path_f, 'model_dir': model_dir_f,
                     'loss.args.unlabeled_weight': unlabeled_weight,
                     'run_name': exp_name, 'group_name': 'landcover',
                     'seed': seed+111, 'dataset.args.seed': seed,
                     'dataset.args.unlabeled_targets_path': pseudolabel_path})

    python_cmd += get_python_cmd(kwargs_f)
    if not get_cmd_only:
        run_sbatch(python_cmd, job_name=exp_name, output=exp_name)
    else:
        return python_cmd


def run_innout_iterated(iterations=2):
    exp_type = 'in-n-out'
    config_path_f = INNOUT_ROOT_PARENT / 'configs/landcover/CNN1DSelfTrainInput.yaml'
    unlabeled_weight = 0.5

    do_pseudolabels = True

    kwargs = {'overwrite': args.overwrite,
              'no_wandb': args.no_wandb,
              'return_best': True,
              'dataset.args.unlabeled_prop': args.unlabeled_prop,
              'restart_epoch_count': True,
              'epochs': 400}

    # TODO can we use 0.05 for all the learning rates
    # kwargs['optimizer.args.lr'] = 0.1
    # kwargs['scheduler.args.lr'] = 0.1
    # # XXX hack - trial 5 was going to nan
    # if args.trial == 5:
    #     kwargs['optimizer.args.lr'] = 0.05
    #     kwargs['scheduler.args.lr'] = 0.05

    kwargs['optimizer.args.lr'] = 0.05
    kwargs['scheduler.args.lr'] = 0.05

    base_exp_id = f'{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{args.trial}'

    cl_extra_deps = []

    trial_cmd = ""
    for st_iteration in range(iterations):
        if st_iteration == 0:
            aux_inputs_name = f'landcover_aux-inputs_unlabeledprop{args.unlabeled_prop}_trial{args.trial}'
            aux_outputs_name = f'landcover_aux-outputs_unlabeledprop{args.unlabeled_prop}_trial{args.trial}_pretrain'
            model_dir_g = model_dir_root / aux_inputs_name
            checkpoint_path_f = model_dir_root / aux_outputs_name / 'best-checkpoint.pt'

            cl_extra_deps.append(aux_inputs_name)
            cl_extra_deps.append(aux_outputs_name)
        else:
            prev_exp_id = 'landcover_' + base_exp_id + f'_iter{st_iteration - 1}_unlabeledweight{unlabeled_weight}'
            model_dir_g = model_dir_root / prev_exp_id
            checkpoint_path_f = model_dir_g / 'best-checkpoint.pt'
            # add some regularization for iterated self training
            kwargs['optimizer.args.weight_decay'] = 0.0
            kwargs['model.args.dropout_prob'] = 0.5

        kwargs.update({'checkpoint_path': checkpoint_path_f})

        curr_exp_id = base_exp_id + f'_iter{st_iteration}'
        cmd = run_selftraining_exp(
                curr_exp_id,
                model_dir_g=model_dir_g, config_path_f=config_path_f,
                kwargs_f=kwargs, seed=args.trial, do_pseudolabels=do_pseudolabels,
                unlabeled_weight=unlabeled_weight,
                get_cmd_only=True)
        if len(trial_cmd) > 0:
            trial_cmd += ' && '
        trial_cmd += cmd
    run_sbatch(trial_cmd, job_name=base_exp_id, output=base_exp_id, cl_extra_deps=cl_extra_deps)



def run_std_selftraining_exps(with_z=False):
    exp_type = 'std_selftrain'
    if with_z:
        exp_type = 'std_selftrain_withz'
    config_path_f = INNOUT_ROOT_PARENT / 'configs/landcover/CNN1DSelfTrainInput.yaml'

    kwargs = {'overwrite': args.overwrite,
              'no_wandb': args.no_wandb,
              'return_best': True,
              'dataset.args.unlabeled_prop': args.unlabeled_prop,
              'restart_epoch_count': True,
              'optimizer.args.lr': 0.1,
              'scheduler.args.lr': 0.1,
              }

    if with_z:
        kwargs['model.args.in_channels'] = 14
        kwargs['dataset.args.include_ERA5'] = True
    aux_inputs_name = f'landcover_aux-inputs_unlabeledprop{args.unlabeled_prop}_trial{args.trial}'
    model_dir_g = model_dir_root / aux_inputs_name

    exp_name = f'{exp_type}_unlabeledprop{args.unlabeled_prop}_trial{args.trial}'
    cmd = run_selftraining_exp(
            exp_name,
            model_dir_g=model_dir_g, config_path_f=config_path_f,
            kwargs_f=kwargs, seed=args.trial, do_pseudolabels=True,
            unlabeled_weight=args.unlabeled_weight,
            get_cmd_only=True)
    run_sbatch(cmd, job_name=exp_name, output=exp_name, cl_extra_deps=[aux_inputs_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts for landcover')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='no wandb')
    parser.add_argument('--dryrun', action='store_true', default=False, help='dryrun')
    parser.add_argument('--unlabeled_prop', type=float, default=0.9,
                        help='proportion of unlabeled data')
    parser.add_argument('--trial', type=int, default=1,
                        help='trial number, also sets the data split seed')
    parser.add_argument('--standardize_unlabeled_sample_size', action='store_true', default=False,
                        help='make unlabeled data size equal across selftrain exps')
    parser.add_argument('--mode', type=str, default='baseline',
                        help='which experiment to run (baseline, aux-input, aux-output, in-n-out)')
    parser.add_argument('--use_slurm', action='store_true', default=False, help='use slurm')
    parser.add_argument('--use_cl', action='store_true', default=False, help='use codalab')
    parser.add_argument('--slurm_partition', type=str, default='jag-standard',
                        help='which slurm partition to use')
    parser.add_argument('--slurm_exclude', type=str, default='jagupard[4-8],jagupard[26-29]',
                        help='which nodes to exclude on slurm')
    parser.add_argument('--unlabeled_weight', type=float, default=0.5,
                        help='weight on unlabeled data for standard self-training')

    args = parser.parse_args()

    if args.use_cl:
        INNOUT_ROOT_PARENT = Path('.')
        INNOUT_ROOT = Path('innout')

    model_subdir = f'landcover_unlabeledprop_{args.unlabeled_prop}'
    if args.standardize_unlabeled_sample_size:
        model_subdir += "_standardizeunlabeled"

    model_dir_root = INNOUT_ROOT_PARENT / 'models' / model_subdir

    if not args.use_cl:
        model_dir_root.mkdir(exist_ok=True, parents=True)

    seed = 1111
    splits = ['train', 'val', 'test']
    root = INNOUT_ROOT_PARENT / 'data'

    if args.mode == 'baseline':
        run_base_exps(aux_inputs=False)
    elif args.mode == 'aux-inputs':
        run_base_exps(aux_inputs=True)
    elif args.mode == 'aux-outputs':
        run_pretrain_exps()
    elif args.mode == 'aux-outputs-onlyunlabeledid':
        run_pretrain_exps(use_unlabeled_ood=False)
    elif args.mode == 'aux-outputs-onlyunlabeledood':
        run_pretrain_exps(use_unlabeled_id=False)
    elif args.mode == 'in-n-out':
        run_innout_iterated(iterations=2)
    elif args.mode == 'standard_selftrain':
        run_std_selftraining_exps()
    elif args.mode == 'standard_selftrain_withz':
        run_std_selftraining_exps(with_z=True)
    else:
        raise ValueError("not implemented")
