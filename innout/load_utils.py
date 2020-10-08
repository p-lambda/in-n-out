import yaml
import importlib
import inspect
from collections import defaultdict
from copy import deepcopy
import ast
from torch.utils.data import DataLoader
from torchvision import transforms


def initialize(obj_config, update_args=None):
    classname = obj_config['classname']
    kwargs = obj_config.get('args')
    if kwargs is None:
        kwargs = {}
    if update_args is not None:
        kwargs.update(update_args)
    return initialize_obj(classname, kwargs)


def get_params(m):
    if hasattr(m, "trainable_params"):
        # "trainable_params" is custom module function
        return m.trainable_params()
    return m.parameters()


def update_config(unparsed, config):
    # handle unknown arguments that change yaml config components
    for unparsed_option in unparsed:
        option_name, val = unparsed_option.split('=')
        # get rid of --
        option_name = option_name[2:].strip()
        # handle nesting
        option_name_list = option_name.split('.')

        # interpret the string as int, float, string, bool, etc
        try:
            val = ast.literal_eval(val.strip())
        except Exception:
            # keep as string
            val = val.strip()

        curr_dict = config
        for k in option_name_list[:-1]:
            try:
                curr_dict = curr_dict.get(k)
            except:
                raise ValueError(f"Dynamic argparse failed: Keys: {option_name_list} Dict: {config}")
        curr_dict[option_name_list[-1]] = val
    return config


def initialize_obj(classname, args_dict=None):
    module_name, class_name = classname.rsplit(".", 1)
    Class = getattr(importlib.import_module(module_name), class_name)
    # filter by argnames
    if args_dict is not None:
        argspec = inspect.getfullargspec(Class.__init__)
        argnames = argspec.args
        args_dict = {k: v for k, v in args_dict.items()
                     if k in argnames or argspec.varkw is not None}

        defaults = argspec.defaults
        # add defaults
        if defaults is not None:
            for argname, default in zip(argnames[-len(defaults):], defaults):
                if argname not in args_dict:
                    args_dict[argname] = default
        class_instance = Class(**args_dict)
    else:
        class_instance = Class()
    return class_instance


def load_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return load_from_config_dict(config)


def load_from_config_dict(config_dict):
    experiments_dict = defaultdict(list)

    for experiment_group in config_dict.keys():
        try:
            # if the keys are classnames
            for experiment_template in config_dict[experiment_group]:
                classname = list(experiment_template.keys())[0]
                experiment_template[classname]['group'] = experiment_group
                exp = initialize_obj(
                    classname,
                    experiment_template[classname])
                experiments_dict[experiment_group].append(exp)
        except AttributeError:
            experiments_dict[experiment_group] = deepcopy(config_dict[experiment_group])
        except Exception as e:
            raise e
    return experiments_dict


def init_transform(config, transform_type):
    '''
    Initializes a PyTorch transform from a config file.

    Parameters
    ----------
    config : dict
        Dictionary representation of .yaml config file.
    transform_type: str
        One of 'train[_target]', 'eval_train[_target]', 'val[_target]', or
        'test[_target]'.

    Returns
    -------
    torchvision.Transform
    '''
    if transform_type + '_transforms' not in config:
        return None

    config_transforms = config[transform_type + '_transforms']
    transform_list = [initialize(trans) for trans in config_transforms]
    return transforms.Compose(transform_list)


def init_dataset(config, dataset_type, template_dataset=None):
    '''
    Initializes a PyTorch Dataset for train, eval_train, validation, or test.

    A few notes:
        - 'train' and 'eval_train' use 'train_transforms'.
        - 'val' and 'test' use 'test_transforms'.
        - 'eval_train' defaults to args in 'train_args' and then updates using
          args in 'eval_train_args'.
        - if config['dataset']['args']['standardize'] is True, then
            - if dataset_type is 'train', then the mean/std of the training
              set is saved in the config file after loading.
            - otherwise, the saved mean/std of the training set from the config
              is used to overwrite the mean/std of the current Dataset.
          Hence, it's important for standardization that the training set is
          first loaded before eval_train/val/test.

    Parameters
    ----------
    config : dict
        Dictionary representation of .yaml config file.
    dataset_type : str
        Either 'train', 'eval-train', 'val', or 'test'.
    template_dataset : torch.utils.data.Dataset, default None
        Optional Dataset to use for initialization.

    Returns
    -------
    torch.utils.data.Dataset
    '''
    custom_type = False
    if dataset_type not in ['train', 'eval_train', 'val', 'test', 'test2']:
        custom_type = True

    transform_type = dataset_type
    if dataset_type in {'eval_train', 'val', 'test2'} or custom_type:
        transform_type = 'test'  # Use test transforms for eval sets.
    transform = init_transform(config, transform_type)
    target_transform = init_transform(config, transform_type + '_target')

    split_type = dataset_type
    if dataset_type == 'eval_train':
        split_type = 'train'  # Default eval_train split is 'train'.
    dataset_kwargs = {'split': split_type, 'transform': transform,
                      'target_transform': target_transform,
                      'template_dataset': template_dataset,
                      'eval_mode': (dataset_type != 'train')}

    if dataset_type == 'eval_train':  # Start off with args in 'train_args'.
        dataset_kwargs.update(config['dataset'].get('train_args', {}))
    dataset_kwargs.update(config['dataset'].get(dataset_type + '_args', {}))

    # We make a copy since the initialize function calls dict.update().
    dataset_config = deepcopy(config['dataset'])
    dataset = initialize(dataset_config, dataset_kwargs)

    if config['dataset'].get('args', {}).get('standardize'):
        if dataset_type == 'train':  # Save training set's mean/std.
            config['dataset']['mean'] = dataset.get_mean()
            config['dataset']['std'] = dataset.get_std()
        else:  # Update dataset with training set's mean and std.
            dataset.set_mean(config['dataset']['mean'])
            dataset.set_std(config['dataset']['std'])

    if config['dataset'].get('args', {}).get('standardize_output'):
        if dataset_type == 'train':  # Save training set's output mean/std.
            config['dataset']['output_mean'] = dataset.get_output_mean()
            config['dataset']['output_std'] = dataset.get_output_std()
        else:  # Update dataset with training set's output mean and std.
            dataset.set_output_mean(config['dataset']['output_mean'])
            dataset.set_output_std(config['dataset']['output_std'])

    return dataset


def init_dataloader(config, dataset, dataset_type, shuffle=True):
    '''
    Initializes a PyTorch DataLoader around a provided dataset. Allows for
    specifying additional arguments via a config file, such as specifying the
    Sampler to use.

    Parameters
    ----------
    config : dict
        Dictionary representation of a .yaml config.
    dataset : torch.utils.data.Dataset
        The PyTorch Dataset to wrap the DataLoader around.
    dataset_type : str
        Either 'train', 'eval_train', 'val', or 'test'.
    '''
    if dataset_type not in ['train', 'eval_train', 'val', 'test', 'test2']:
        raise ValueError('{} is an invalid dataset type!'.format(dataset_type))
    dl_kwargs = {}
    if config['use_cuda']:
        dl_kwargs = {'num_workers': 2, 'pin_memory': True}
    batch_size = config.get('batch_size', 256)
    if dataset_type != 'train':
        batch_size = config.get('eval_batch_size', 256)
    dl_kwargs = {'batch_size': batch_size, 'shuffle': shuffle}

    if 'dataloader' in config:
        sampler = None
        dataloader_args = config['dataloader'].get('args', {})
        if 'sampler' in dataloader_args:
            sampler_kwargs = {'data_source': dataset}
            sampler = initialize(dataloader_args['sampler'], sampler_kwargs)
        dl_kwargs.update(dataloader_args)
        dataloader_args = config['dataloader'].get(dataset_type + '_args', {})
        if 'sampler' in dataloader_args:
            sampler_kwargs = {'data_source': dataset}
            sampler = initialize(dataloader_args['sampler'], sampler_kwargs)
        dl_kwargs.update(dataloader_args)
        dl_kwargs['sampler'] = sampler

    return DataLoader(dataset, **dl_kwargs)
