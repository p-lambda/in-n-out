import subprocess
import shlex
import argparse
import json
import torch
import numpy as n
from copy import deepcopy
from pathlib import Path
import datetime
import pandas as pd

# from extrapolation.configs.celeba.gender_in_no_hats_out_hats.two_stage_attr_p2 import get_input_results, get_outputs_attributes_labels

WORKSHEET_NAME = 'in-n-out-iclr'
DOCKER_IMAGE = 'ananya/in-n-out'


def get_time_str():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    return dt_string


def get_python_cmd(kwargs=None, python_path='python', code_path='code/'):
    if kwargs is not None:
        opts = ''.join([f"--{k}={v} " for k, v in kwargs.items() if not isinstance(v, bool) or '.' in k])
        opts += ''.join([f"--{k} " for k, v in kwargs.items() if isinstance(v, bool) and v and '.' not in k])
    else:
        opts = ''
    python_cmd = python_path + ' ' +\
                 code_path + '/main.py '
    python_cmd += opts
    return python_cmd


def run_sbatch(python_cmd, sbatch_script_path, job_name='innout', output='logs', mail_type='END,FAIL',
               mail_user='eix@cs.stanford.edu', partition='jag-standard', exclude='jagupard[4-8,10-11]',
               nodes=1, gres='gpu:1', cpus_per_task=1, mem='16G'):

    cmd = f'sbatch --job-name={job_name} --output={output} --mail-type={mail_type} --mail-user={mail_user} ' +\
          f'--partition={partition} --exclude={exclude} --nodes={nodes} ' +\
          f'--gres={gres} --cpus-per-task={cpus_per_task} --mem={mem} {sbatch_script_path} '

    cmd += f'"{python_cmd}"'
    print(cmd)
    subprocess.run(shlex.split(cmd))


def run_codalab(python_cmd, job_name, gpus=1, mem='16G', cpus=1, nlp=True):
    prefix = (f'cl run -n {job_name} -w {WORKSHEET_NAME} --request-docker-image={DOCKER_IMAGE} '
              f'--request-gpus={gpus} --request-memory={mem} --request-cpus={cpus} ')
    if nlp:
        nlp_opt = '--request-queue tag=nlp '
    else:
        nlp_opt = ''
    bundles = ':innout :configs :celeba_pickle '
    makedirs = '"export PYTHONPATH="."; mkdir logs; mkdir outputs; '
    codalab_cmd = prefix + nlp_opt + bundles + makedirs + python_cmd + '"'
    print(codalab_cmd)
    subprocess.run(shlex.split(codalab_cmd))


def run_job_chain(kwargs_list, job_name, output, sbatch_script_path, python_path, code_path,
                  codalab=False, mail_type='END,FAIL', mail_user=None, partition='jag-standard',
                  exclude='jagupard[4-8,10-11]', nodes=1, gres='gpu:1', cpus_per_task=1,
                  mem='16G'):
    '''
    kwargs_list: list of dict
        list of kwargs
    '''
    cmd = ''
    for kwargs in kwargs_list:
        python_cmd = get_python_cmd(kwargs, python_path=python_path, code_path=code_path)
        if len(cmd) > 0:
            cmd += ' && '
        if codalab:
            cmd += python_cmd + ' --no_wandb '
        else:
            cmd += python_cmd
    if codalab:
        run_codalab(cmd, job_name=job_name, gpus=int(gres[gres.rfind(':')+1:]),
                    cpus=cpus_per_task, mem=mem, nlp=True)
    else:
        run_sbatch(cmd, sbatch_script_path=sbatch_script_path, job_name=job_name, output=output,
                   mail_type=mail_type, mail_user=mail_user, partition=partition, exclude=exclude,
                   nodes=nodes, gres=gres, cpus_per_task=cpus_per_task, mem=mem) 


def run_input_exp():
    kwargs = {}
    kwargs = deepcopy(kwargs)
    print(exp_name)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_input',
                   'seed': seed, 'project_name': 'extrapolation',
                   'dataset.args.seed': seed})
    kwargs_list = []
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs_list.append(kwargs.copy())


def run_input_experiments():
    for i in range(num_trials):
        run_input_exp(
            exp_name='input_celeba_' + str(args.lr) + '_trial' + str(i),
            model_dir=args.log_dir + '/celeba_input_' + str(args.lr) + '_trial' + str(i),
            config_path=args.config_dir + 'input_model_p1.yaml',
            seed=i,
            lr=args.lr,
            args=args)


def run_pretrain_exp(exp_name, model_dir, config_path, freeze_shared=False, seed=0, lr=0.03, args=None):
    kwargs = {}
    kwargs = deepcopy(kwargs)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_output',
                   'seed': seed, 'project_name': 'extrapolation',
                   'dataset.args.seed': seed})

    kwargs_list = []
    kwargs['model_dir'] = str(model_dir) + '/pretrain/'
    kwargs['run_name'] = exp_name + '_pretrain'
    kwargs['dataset.train_args.split'] = 'all_unlabeled'
    kwargs['dataset.args.pickle_file_path'] = args.celeba_pickle_dir + '/celeba_train_pickle'
    kwargs['dataset.args.celeba_root'] = args.celeba_pickle_dir
    kwargs_list.append(kwargs.copy())

    # resume from previous checkpoint
    kwargs['checkpoint_path'] = str(Path(kwargs['model_dir']) / 'best-checkpoint.pt')
    kwargs['dataset.train_args.split'] = 'train'
    kwargs['model_dir'] = str(model_dir) + '/finetune/'
    kwargs['run_name'] = exp_name + '_finetune'
    kwargs['loss.classname'] = 'torch.nn.BCEWithLogitsLoss'
    kwargs['eval_loss.classname'] = 'torch.nn.BCEWithLogitsLoss'
    kwargs['model.args.use_idx'] = 0
    kwargs['restart_epoch_count'] = True
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs['dataset.args.meta_as_target'] = False
    kwargs['dataset.args.pickle_file_path'] = args.celeba_pickle_dir + '/celeba_train_pickle'
    kwargs['dataset.args.celeba_root'] = args.celeba_pickle_dir
    kwargs_list.append(kwargs.copy())
    run_job_chain(
            kwargs_list=kwargs_list, python_path=args.python_path, code_path=args.code_path,
            job_name=exp_name, output=args.output_dir + f'/{exp_name}',
            mail_user=args.mail_user, codalab=args.codalab,
            sbatch_script_path=args.sbatch_script_path)


def run_pretrain_experiments(args):
    for i in range(args.num_trials):
        run_pretrain_exp(
            exp_name='output_celeba_' + str(args.lr) + '_trial' + str(i),
            model_dir=args.log_dir + '/celeba_output_' + str(args.lr) + '_trial' + str(i),
            config_path=args.config_dir + 'output_model_p1.yaml',
            seed=i,
            lr=args.lr,
            args=args)


def run_baseline_exp(exp_name, model_dir, config_path, seed, lr, args):
    kwargs = {}
    kwargs = deepcopy(kwargs)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_baseline',
                   'seed': seed, 'project_name': 'innout',
                   'dataset.args.seed': seed})
    kwargs_list = []
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs['dataset.args.pickle_file_path'] = args.celeba_pickle_dir + '/celeba_train_pickle'
    kwargs['dataset.args.celeba_root'] = args.celeba_pickle_dir
    kwargs_list.append(kwargs.copy())
    run_job_chain(
            kwargs_list=kwargs_list, python_path=args.python_path, code_path=args.code_path,
            job_name=exp_name, output=args.output_dir + f'/{exp_name}',
            mail_user=args.mail_user, codalab=args.codalab,
            sbatch_script_path=args.sbatch_script_path)


def run_baseline_experiments(args):
    for i in range(args.num_trials):
        run_baseline_exp(
            exp_name='baseline_celeba_' + str(args.lr) + '_trial' + str(i),
            model_dir=args.log_dir + '/celeba_baseline_' + str(args.lr) + '_trial' + str(i),
            config_path=args.config_dir + 'baseline_resnet.yaml',
            seed=i,
            lr=args.lr,
            args=args)
    

def main(args):
    if args.experiment == 'baseline':
        run_baseline_experiments(args)
    if args.experiment == 'output':
        run_pretrain_experiments(args)
    if args.expepriment == 'input':
        run_input_experiments(args)
    # run_baseline_experiments(lr=0.03)
    
    # run_pretrain_experiments()
    # run_pretrain_experiments(lr=0.1)
    # run_input_experiments_p1()
    # run_input_experiments_p2()
    # run_selftraining_exps(unlabeled_weight=0.5)
    # for ulw in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    #     run_selftraining_exps(unlabeled_weight=ulw)
    # for ulw in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        # run_selftraining_exps(unlabeled_weight=ulw, pretrain=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run celeba experiments.')
    parser.add_argument('--codalab', action='store_true', help='run on CodaLab not slurm')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment to run.')
    parser.add_argument('--num_trials', type=int, required=True,
                        help='Number of trials to run of the experiment.')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate to use when training model.')
    parser.add_argument('--python_path', type=str, required=False,
                        help='Path or alias to Python interpreter', default='python')
    parser.add_argument('--code_path', type=str, required=False,
                        help='Path to directory where main.py is located.', default='innout/')
    parser.add_argument('--config_dir', type=str, required=False,
                        help='Directory where config files are stored.', default='configs/celeba/')
    parser.add_argument('--log_dir', type=str, required=False,
                        help='Path to save logs and checkpoints.', default='logs/')
    parser.add_argument('--output_dir', type=str, required=False,
                        help='Path to dir to store stdout for experiment.', default='outputs/')
    parser.add_argument('--celeba_pickle_dir', type=str, required=False,
                        help='Location of celeb-a pickle file.', default='celeba_pickle')
    parser.add_argument('--mail_user', type=str, required=False,
                        help='(Slurm only, optional) Email if slurm job fails.', default=None)
    parser.add_argument('--sbatch_script_path', type=str, required=False,
                        help='(Slurm only, required for slurm) sbatch script', default=None)
    args, unparsed = parser.parse_known_args()

    main(args)
    
