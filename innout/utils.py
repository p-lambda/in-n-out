"""
Some utilities
"""

import pandas as pd
import datetime
import torch


class DataParallel(torch.nn.DataParallel):
    '''
    Pass-through the attributes of the model thru DataParallel
    '''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def to_device(obj, device):
    '''
    Wrapper around torch.Tensor.to that handles the case when obj is a
    container.

    Parameters
    ----------
    obj : Union[torch.Tensor, List[torch.Tensor], Dict[Any, Any]]
        Object to move to the specified device.
    device : str
        Describes device to move to.

    Returns
    -------
    Same type as obj.
    '''
    if isinstance(obj, list):
        return [item.to(device) for item in obj]
    elif isinstance(obj, dict):
        res = {}
        for key in obj:
            value = obj[key]
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            res[key] = value
        return res
    else:
        return obj.to(device)


def time_to_str(ts):
    return pd.Timestamp(ts).strftime('%Y-%m-%dT%H-%M-%S-%f')


def now_to_str():
    return time_to_str(datetime.datetime.now())


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def readlines(path):
    with open(path) as f:
        lines = f.readlines()
    return lines

def writelines(lines, path):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)

def dict_to_cli(args_dict):
    '''
    Convert a dictionary to an args list expected by argparse. Use only the full name of the argument as keys in the dictionary (--foo instead of -f).
    '''
    # return [f'--{k}={v}' for k, v in args_dict.items()]
    args = []
    for k, v in args_dict.items():
        if v is not None:
            args.append(f'--{k}')
            args.append(str(v))
        else:
            args.append(str(k))
    return args
