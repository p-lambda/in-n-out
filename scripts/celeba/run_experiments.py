import subprocess
import shlex
import argparse
import json
import numpy as np
from copy import deepcopy
from pathlib import Path
import datetime
import pandas as pd

# from extrapolation.configs.celeba.gender_in_no_hats_out_hats.two_stage_attr_p2 import get_input_results, get_outputs_attributes_labels

WORKSHEET_NAME = 'skywalker94-in-n-out-celeba'
# WORKSHEET_NAME = 'in-n-out-iclr'
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


def run_codalab(python_cmd, job_name, gpus=1, mem='16G', cpus=1, nlp=True, extra_cl_deps=''):
    prefix = (f'cl run -n {job_name} -w {WORKSHEET_NAME} --request-docker-image={DOCKER_IMAGE} '
              f'--request-gpus={gpus} --request-memory={mem} --request-cpus={cpus} ')
    if nlp:
        nlp_opt = '--request-queue tag=nlp '
    else:
        nlp_opt = ''
    bundles = ':innout :configs :celeba_pickle :scripts ' + extra_cl_deps + ' '
    makedirs = '"export PYTHONPATH="."; mkdir logs; mkdir outputs; '
    codalab_cmd = prefix + nlp_opt + bundles + makedirs + python_cmd + '"'
    print(codalab_cmd)
    subprocess.run(shlex.split(codalab_cmd))


def get_chain_python_cmd(kwargs_list, python_path, code_path, codalab=False):
    cmd = ''
    for kwargs in kwargs_list:
        python_cmd = get_python_cmd(kwargs, python_path=python_path, code_path=code_path)
        if len(cmd) > 0:
            cmd += ' && '
        if codalab:
            cmd += python_cmd + ' --no_wandb '
        else:
            cmd += python_cmd
    return cmd


def run_job(cmd, job_name, output, sbatch_script_path, codalab=False, mail_type='END,FAIL',
            mail_user=None, partition='jag-standard', exclude='jagupard[4-8,10-11]',
            nodes=1, gres='gpu:1', cpus_per_task=1, mem='16G', extra_cl_deps=''):
    if codalab:
        run_codalab(cmd, job_name=job_name, gpus=int(gres[gres.rfind(':')+1:]),
                    cpus=cpus_per_task, mem=mem, nlp=True, extra_cl_deps=extra_cl_deps)
    else:
        run_sbatch(cmd, sbatch_script_path=sbatch_script_path, job_name=job_name, output=output,
                   mail_type=mail_type, mail_user=mail_user, partition=partition, exclude=exclude,
                   nodes=nodes, gres=gres, cpus_per_task=cpus_per_task, mem=mem) 


def run_input_exp(exp_name, model_dir, config_path, seed=0, lr=0.1, args=None):
    kwargs = {}
    kwargs = deepcopy(kwargs)
    print(exp_name)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_input',
                   'seed': seed, 'project_name': 'innout',
                   'dataset.args.seed': seed})
    kwargs_list = []
    kwargs['model_dir'] = str(model_dir)
    kwargs['run_name'] = exp_name + '_input_p1'
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs['dataset.args.pickle_file_path'] = args.celeba_pickle_dir + '/celeba_train_pickle'
    kwargs['dataset.args.celeba_root'] = args.celeba_pickle_dir
    kwargs_list.append(kwargs.copy())
    cmd = get_chain_python_cmd(
            kwargs_list=kwargs_list, python_path=args.python_path, code_path=args.code_path,
            codalab=args.codalab)
    cmd += ' && mkdir -p ' + args.pseudolabels_dir
    cmd += (' && ' + args.python_path + ' ' + args.scripts_dir + '/celeba/input_sklearn_p2.py' +
            ' --model_dir=' + model_dir + ' --pseudolabels_dir=' + args.pseudolabels_dir + ' ')
    run_job(cmd,
            job_name=exp_name, output=args.output_dir + f'/{exp_name}',
            mail_user=args.mail_user, codalab=args.codalab,
            sbatch_script_path=args.sbatch_script_path)


def run_input_experiments(args):
    for i in range(args.num_trials):
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
                   'seed': seed, 'project_name': 'innout',
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
    cmd = get_chain_python_cmd(
            kwargs_list=kwargs_list, python_path=args.python_path, code_path=args.code_path,
            codalab=args.codalab)
    run_job(cmd,
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
    cmd = get_chain_python_cmd(
            kwargs_list=kwargs_list, python_path=args.python_path, code_path=args.code_path,
            codalab=args.codalab)
    run_job(
            cmd,
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
    
def run_selftrain_exp(exp_name, model_dir, config_path, seed, lr, 
                      pseudolabeler_dir, init_dir, use_ood_unlabeled, dropout,
                      unlabeled_weight, args):
    # Generate pseudolabels.
    pseudolabels_dir = 'pseudolabels'
    pseudolabeler_bundle = pseudolabeler_dir[:pseudolabeler_dir.find('/')]
    init_bundle = init_dir[:init_dir.find('/')]
    extra_cl_deps = ' :' + pseudolabeler_bundle.strip() + ' :' + init_bundle.strip()
    mkdir_cmd = ' mkdir -p ' + pseudolabels_dir
    cp_cmd = ' cp -r ' + pseudolabeler_dir + '/* pseudolabels '
    ood_unlabeled_opt = ' --use_unlabeled_ood ' if use_ood_unlabeled else ' '
    python_cmd = (args.python_path + ' ' + args.scripts_dir + '/get_pseudolabels.py ' + 
                  '--model_dir=' + pseudolabels_dir + ' --use_unlabeled_id ' +
                  ood_unlabeled_opt + ' --is_binary --pseudolabel_path=' + pseudolabels_dir +
                  '/pseudolabels.npy')
    pseudolabels_cmd = mkdir_cmd + ' && ' + cp_cmd + ' && ' + python_cmd
    # Train on pseudolabels.
    kwargs = {}
    kwargs = deepcopy(kwargs)
    kwargs.update({'config': config_path, 'model_dir': model_dir,
                   'run_name': exp_name, 'group_name': 'celeba_innout',
                   'seed': seed, 'project_name': 'innout',
                   'dataset.args.seed': seed})
    kwargs_list = []
    # resume from previous checkpoint
    kwargs['checkpoint_path'] = str(Path(init_dir) / 'best-checkpoint.pt')
    kwargs['dataset.train_args.split'] = 'train'
    if use_ood_unlabeled:
        kwargs['dataset.train_args.use_unlabeled_ood'] = True
    kwargs['dataset.train_args.use_unlabeled_id'] = True
    kwargs['dataset.train_args.unlabeled_target_path'] = pseudolabels_dir + '/pseudolabels.npy'
    kwargs['loss.args.unlabeled_weight'] = unlabeled_weight
    kwargs['model_dir'] = str(model_dir)
    kwargs['run_name'] = exp_name
    kwargs['model.args.use_idx'] = 0
    kwargs['restart_epoch_count'] = True
    kwargs['optimizer.args.lr'] = lr
    kwargs['scheduler.args.lr'] = kwargs['optimizer.args.lr']
    kwargs['dataset.args.meta_as_target'] = False
    kwargs['dataset.args.pickle_file_path'] = args.celeba_pickle_dir + '/celeba_train_pickle'
    kwargs['dataset.args.celeba_root'] = args.celeba_pickle_dir
    kwargs['model.args.dropout_prob'] = dropout
    kwargs_list.append(kwargs.copy())
    self_train_cmd = get_chain_python_cmd(
        kwargs_list=kwargs_list, python_path=args.python_path, code_path=args.code_path,
        codalab=args.codalab)
    cmd = pseudolabels_cmd + ' && ' + self_train_cmd
    run_job(
            cmd,
            job_name=exp_name, output=args.output_dir + f'/{exp_name}',
            mail_user=args.mail_user, codalab=args.codalab,
            sbatch_script_path=args.sbatch_script_path,
            extra_cl_deps=extra_cl_deps)
 
def run_selftrain_experiments(args):
    for i in range(args.num_trials):
        exp_name = ('selftrain_round' + str(args.st_round) +
                    '_celeba_' + str(args.lr) + '_trial' + str(i))
        model_dir = args.log_dir + '/' + exp_name
        pseudolabeler_dir = (args.pseudolab_dir_base + str(i) + '/' +
                             args.pseudolab_model_dir_base + str(i) + '/')
        init_dir = (args.init_dir_base + str(i) + '/' +
                    args.init_model_dir_base + str(i) + '/')
        run_selftrain_exp(
            exp_name=exp_name,
            model_dir=model_dir,
            config_path=args.config_dir + 'innout.yaml',
            seed=i,
            lr=args.lr,
            pseudolabeler_dir=pseudolabeler_dir,
            init_dir=init_dir,
            use_ood_unlabeled=args.use_ood_unlabeled,
            dropout=args.dropout,
            unlabeled_weight=args.unlabeled_weight,
            args=args)


def main(args):
    if args.experiment == 'baseline':
        run_baseline_experiments(args)
    if args.experiment == 'output':
        run_pretrain_experiments(args)
    if args.experiment == 'input':
        run_input_experiments(args)
    if args.experiment == 'self-train':
        run_selftrain_experiments(args)
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
    parser.add_argument('--scripts_dir', type=str, required=False,
                        help='Path to dir where scripts are stored.', default='scripts/')
    parser.add_argument('--pseudolabels_dir', type=str, required=False,
                        help='Path to dir to store pseudolabels for aux-in.', default='pseudolabels/')
    parser.add_argument('--pseudolab_dir_base', type=str, required=False,
                        help='Directory for the pseudolabeler model.')
    parser.add_argument('--pseudolab_model_dir_base', type=str, required=False,
                        help='Path to dir where the pseudolabeler model config is stored.')
    parser.add_argument('--init_dir_base', type=str, required=False,
                        help='Directory for the model to initialize from.')
    parser.add_argument('--init_model_dir_base', type=str, required=False,
                        help='Path to dir where the initializing model config is stored.')
    parser.add_argument('--st_round', type=int, required=False,
                        help='Round of self-training.')
    parser.add_argument('--dropout', type=float, required=False,
                        help='How much dropout to add for self-training.')
    parser.add_argument('--unlabeled_weight', type=float, required=False,
                        help='How much to weight the unlabeled data.')
    parser.add_argument('--use_ood_unlabeled', action='store_true',
                        help='Use OOD unlabeled data when self-training.')
    parser.add_argument('--celeba_pickle_dir', type=str, required=False,
                        help='Location of celeb-a pickle file.', default='celeba_pickle')
    parser.add_argument('--mail_user', type=str, required=False,
                        help='(Slurm only, optional) Email if slurm job fails.', default=None)
    parser.add_argument('--sbatch_script_path', type=str, required=False,
                        help='(Slurm only, required for slurm) sbatch script', default=None)
    
    args, unparsed = parser.parse_known_args()

    main(args)
    
