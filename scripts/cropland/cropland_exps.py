import argparse
from pathlib import Path
import subprocess
import shlex
from copy import deepcopy

from innout import INNOUT_ROOT
INNOUT_ROOT_PARENT = INNOUT_ROOT.parent

BASELINE_CONFIG = 'configs/cropland/unet_prediction_indianakentuckyood.yaml'
PRETRAIN_CONFIG = 'configs/cropland/unet_pretrain_indianakentuckyood.yaml'
INNOUT_CONFIG = 'configs/cropland/unet_selftrain.yaml'

EXPERIMENT_MODES = ['baseline', 'aux-inputs', 'aux-outputs', 'in-n-out']
DEFAULT_SEED = 65269
NUM_INNOUT_ITERATIONS = 2

CODALAB_WORKSHEET = 'in-n-out-iclr-cropland'

def get_python_cmd(args, main_args={}):
    '''
    Constructs the Python command to invoke main.py and run the experiment.
    Formats any additional arguments to main.py.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    main_args : Dict[str, Any], default {}
        Extra arguments to main.py.

    Returns
    -------
    python_cmd : str
        Python command to run experiment.
    '''
    # Arbitrary way of setting seed based on trial number, not super important
    seed = DEFAULT_SEED + (args.trial_num - 1) * 111
    
    # Set some arguments for main.py that aren't in config
    main_args = deepcopy(main_args)
    main_args['overwrite'] = args.overwrite
    main_args['no_wandb'] = args.no_wandb
    main_args['group_name'] = 'cropland'
    main_args['return_best'] = True
    main_args['seed'] = seed
    main_args['dataset.args.seed'] = seed + 1

    opts = ''.join([f"--{k}={v} " for k, v in main_args.items() if not isinstance(v, bool) or '.' in k])
    opts += ''.join([f"--{k} " for k, v in main_args.items() if isinstance(v, bool) and v and '.' not in k])

    if args.use_cl:
        python_cmd = 'python ' + f'{INNOUT_ROOT}/main.py '
    else:
        python_cmd = f'{INNOUT_ROOT_PARENT}/.env/bin/python ' +\
            f'{INNOUT_ROOT}/main.py '
    python_cmd += opts
    return python_cmd


def run_exp(args, python_cmd, exp_name, cl_extra_deps=None):
    '''
    Executes the python command on the desired platform.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    python_cmd : str
        Python command to run.
    exp_name : str
        Name of the experiment.
    cl_extra_deps : List[str], default None
        Extra dependencies for CodaLab.
    '''
    if args.use_slurm:
        logpath = INNOUT_ROOT_PARENT / 'logs' / exp_name
        logpath.parent.mkdir(exist_ok=True)
        run_sbatch_script = INNOUT_ROOT_PARENT / 'run_sbatch.sh'
        cmd = f'sbatch --job-name={exp_name} --output={logpath} '
        cmd += '--partition=jag-standard --exclude=jagupard[4-8,26-29] '
        cmd += f'--gres=gpu:1 --mem=52GB {run_sbatch_script} '
        cmd += f'"{python_cmd}"'
    elif args.use_cl:
        if cl_extra_deps is not None:
            cl_extra_deps_str = ' '.join([f":{dep}" for dep in cl_extra_deps])
            cl_extra_deps_cps = ' '.join([f"cp -r {dep}/models/cropland/* models/cropland;" for dep in cl_extra_deps if not dep.endswith('.py')])
            cl_extra_deps_cps = 'mkdir models/cropland; ' + cl_extra_deps_cps
        else:
            cl_extra_deps_str = ''
            cl_extra_deps_cps = ''

        cmd = f'cl run -n {exp_name} -w {args.codalab_worksheet} '
        cmd += '--request-docker-image ananya/in-n-out --request-gpus 1 '
        cmd += '--request-memory 52g --request-queue tag=nlp :innout :configs '
        cmd += f':cropland_data.pkl {cl_extra_deps_str} '
        cmd += '"export PYTHONPATH=.; mkdir models; '
        cmd += f'mkdir innout-pseudolabels; {cl_extra_deps_cps} {python_cmd}"'
    else:
        cmd = python_cmd
    print(cmd)
    if not args.dryrun:
        subprocess.run(shlex.split(cmd))


def run_base_exp(args, use_aux_inputs=False):
    '''
    Runs the "baseline" experiment (supervised learning with no auxiliary data)
    or the "aux-inputs" experiment (supervised learning with auxiliary data
    included as features).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    use_aux_inputs : bool, default False
        Whether to include auxiliary data as input features.
    '''
    config_path = INNOUT_ROOT_PARENT / BASELINE_CONFIG
    main_args = {}
    if use_aux_inputs:
        exp_type = 'aux-inputs'
        main_args['dataset.args.in_bands'] = '[1,2,3,10,11,12]'
        main_args['dataset.args.include_lat_lon'] = 'True'
        main_args['model.args.in_channels'] = 8
    else:
        exp_type = 'baseline'

    exp_name = f'cropland_{exp_type}_trial{args.trial_num}'
    model_dir_root = INNOUT_ROOT_PARENT / 'models' / 'cropland'
    if not args.use_cl:
        model_dir_root.mkdir(exist_ok=True, parents=True)
    model_dir = model_dir_root / exp_name
    main_args['config'] = config_path
    main_args['model_dir'] = model_dir
    main_args['run_name'] = exp_name
    python_cmd = get_python_cmd(args, main_args)
    run_exp(args, python_cmd, exp_name)


def run_pretrain_exp(args):
    '''
    Runs the aux-outputs experiment, which uses the auxiliary information as
    targets for unsupervised pretraining. The pretrained model is then
    finetuned on labeled data without incorporating the auxiliary information.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    '''

    exp_name = f'cropland_aux-outputs_trial{args.trial_num}'
    model_dir = INNOUT_ROOT_PARENT / 'models' / 'cropland' / exp_name
    pretrain_config_path = INNOUT_ROOT_PARENT / PRETRAIN_CONFIG
    pretrain_args = {'config': pretrain_config_path,
                     'model_dir': str(model_dir) + '_pretrain',
                     'run_name': exp_name + '_pretrain'}

    finetune_config_path = INNOUT_ROOT_PARENT / BASELINE_CONFIG
    checkpoint_path = pretrain_args['model_dir'] + '/best-checkpoint.pt'
    finetune_args = {'config': finetune_config_path,
                     'model_dir': model_dir,
                     'run_name': exp_name,
                     'restart_epoch_count': True,
                     'checkpoint_path': checkpoint_path}

    main_args_list = [pretrain_args, finetune_args]
    python_cmd = get_python_cmd(args, main_args_list[0])
    python_cmd += ' && ' + get_python_cmd(args, main_args_list[1])
    run_exp(args, python_cmd, exp_name)


def run_innout_iterated(args, num_iterations):
    innout_config_path = INNOUT_ROOT_PARENT / INNOUT_CONFIG
    innout_args = {'config': innout_config_path,
                   'restart_epoch_count': True,
                   'loss.args.unlabeled_weight': 0.5}
    pseudolabel_dir = INNOUT_ROOT_PARENT / 'innout-pseudolabels'
    pseudolabel_script = 'get_pseudolabels.py'
    python_path = 'python'
    if not args.use_cl:
        python_path = INNOUT_ROOT_PARENT / '.env' / 'bin' / python_path
        pseudolabel_script = INNOUT_ROOT_PARENT / 'scripts' / pseudolabel_script
        pseudolabel_dir.mkdir(exist_ok=True, parents=True)
    cl_extra_deps = ['get_pseudolabels.py']
    model_dir = INNOUT_ROOT_PARENT / 'models' / 'cropland'
    run_name_format = 'cropland_in-n-out_iter{}_trial{}'
    python_cmd = ''
    for iteration in range(num_iterations):
        innout_args['dataset.args.use_unlabeled_id'] = True
        if iteration == 0:
            pseudolabel_model_name = f'cropland_aux-inputs_trial{args.trial_num}'
            pretrained_model_name = f'cropland_aux-outputs_trial{args.trial_num}_pretrain'
            innout_args['dataset.args.use_unlabeled_ood'] = False
            cl_extra_deps.append(pseudolabel_model_name)
            cl_extra_deps.append('_'.join(pretrained_model_name.split('_')[:-1]))
        else:
            prev_run_name = run_name_format.format(iteration - 1, args.trial_num)
            pseudolabel_model_name = pretrained_model_name = prev_run_name
            innout_args['model.args.dropout_prop'] = 0.8
            innout_args['dataset.args.use_unlabeled_ood'] = True

        pseudolabel_model_dir = model_dir / pseudolabel_model_name
        pretrained_checkpoint = model_dir / pretrained_model_name / 'best-checkpoint.pt'
        innout_args['checkpoint_path'] = pretrained_checkpoint
        run_name = run_name_format.format(iteration, args.trial_num)
        pseudolabel_path = pseudolabel_dir / f'{run_name}.npy'
        if iteration > 0:
            python_cmd += ' && '
        python_cmd += f'{python_path} {pseudolabel_script} --is_binary'
        python_cmd += f' --model_dir {pseudolabel_model_dir}'
        python_cmd += '  --use_unlabeled_id'
        if innout_args['dataset.args.use_unlabeled_ood']:
            python_cmd += ' --use_unlabeled_ood'
        python_cmd += f' --pseudolabel_path {pseudolabel_path} && '

        innout_args['model_dir'] = model_dir / run_name
        innout_args['dataset.args.unlabeled_targets_path'] = pseudolabel_path
        python_cmd += get_python_cmd(args, innout_args)

    run_exp(args, python_cmd, f'cropland_in-n-out_trial{args.trial_num}', cl_extra_deps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Runs scripts for cropland')
    parser.add_argument('mode', type=str, choices=EXPERIMENT_MODES,
                        help='Type of experiment to run')
    parser.add_argument('--trial_num', type=int, default=1,
                        help='Experiment Trial')
    cluster_group = parser.add_mutually_exclusive_group()
    cluster_group.add_argument('--use_slurm', action='store_true',
                               help='Run experiment on Slurm')
    cluster_group.add_argument('--use_cl', action='store_true',
                               help='Run experiment on CodaLab')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite previous run if same name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Don\'t use W&B')
    parser.add_argument('--dryrun', action='store_true',
                        help='Print command and exit')
    parser.add_argument('--codalab_worksheet', type=str, default=CODALAB_WORKSHEET,
                        help='Codalab worksheet')

    args = parser.parse_args()

    if args.use_cl:
        INNOUT_ROOT_PARENT = Path('.')
        INNOUT_ROOT = Path('innout')

    if args.mode == 'baseline':
        run_base_exp(args, use_aux_inputs=False)
    elif args.mode == 'aux-inputs':
        run_base_exp(args, use_aux_inputs=True)
    elif args.mode == 'aux-outputs':
        run_pretrain_exp(args)
    elif args.mode == 'in-n-out':
        run_innout_iterated(args, num_iterations=NUM_INNOUT_ITERATIONS)
