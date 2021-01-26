from pathlib import Path
import numpy as np
import shutil
import socket


def filter_data_map(data_map, indices):
    '''
    Filters a map storing NumPy arrays based on the given indices. The indexing
    array can either be an array of integers or a mask of booleans. Creates a
    shallow copy of the provided dictionary for filtering.

    Note that while convenient, indexing into NumPy arrays creates deep copies
    of the subarrays. This can quickly lead to OOM errors if the dataset is
    large.

    Parameters
    ----------
    data_map : Dict[Any, Any]
        Dictionary to filter.
    indices : numpy.ndarray[Union[int, bool]]
        Indexing array for filtering.

    Returns
    -------
    Dict[Any, Any]
        Shallow copy of data_map after filtering.
    '''
    data_map = data_map.copy()
    for key in data_map.keys():
        data = data_map[key]
        if isinstance(data, np.ndarray) and data.size > 0:
            data = data[indices]
        data_map[key] = data
    return data_map


def get_local_disk_path():
    '''
    Returns the path to the local disk on the current machine, or None if it
    does not exist. For example, on john10 this would return
    pathlib.Path('/john10'), but on sc this would return None.

    Returns
    -------
    Union[pathlib.Path, None]
        Path to the current machine's local disk or None if it does not exist.
    '''
    machine_name = socket.gethostname().split('.')[0]
    local_disk_path = '/' / Path(machine_name)
    return local_disk_path if local_disk_path.exists() else None


def search_local(sub_path):
    '''
    Attempts to find the specified file on the current machine's local disk.
    Loops through scratch directories (those that start with 'scr') and appends
    sub_path to look for the file. For example, if the current machine is
    jagupard10 and sub_path is 'crops/state_splits/kentucky.pkl', then this
    function will search through

        /jagupard10/scr1/crops/state_splits/kentucky.pkl
        /jagupard10/scr2/crops/state_splits/kentucky.pkl
        ...

    etc. until all scratch directories have been tried or the file is found. If
    the file is not found, None is returned.

    Parameters
    ----------
    sub_path : Union[str, pathlib.Path]
        The path to look for relative to the machine's scratch directories.

    Returns
    -------
    pathlib.Path or None
        The path to the file on the local machine's disk if found, else None.
    '''
    sub_path = Path(sub_path)
    local_disk_path = get_local_disk_path()
    if local_disk_path is None:
        return None

    for dir_path in local_disk_path.iterdir():
        if dir_path.stem.startswith('scr'):
            full_path = dir_path / sub_path
            if full_path.exists():
                return full_path
    return None


def copy_to_local(src_path, dst_sub_path):
    '''
    Copies a file into the scratch directory with the most free space on the
    current machine. The destination path should be relative. For example, if
    this function runs on jagupard15 where scr4 has the most space, src_path is
    /u/nlp/data/landcover/landcover_v2.pkl and dst_sub_path is
    landcover/cache.pkl, then this function will copy landcover_v2.pkl into the
    location /jagupard15/scr4/landcover/cache.pkl, creating the necessary
    parent folders.

    Parameters
    ----------
    src_path : Union[str, pathlib.Path]
        Path to the file to copy over.
    dst_sub_path : Union[str, pathlib.Path]
        Path to the destination file on the local disk (relative to the scatch
        directories).

    Returns
    -------
    pathlib.Path
        Absolute path to the copied file on the local machine.
    '''
    dst_sub_path = Path(dst_sub_path)
    assert not dst_sub_path.is_absolute()
    local_disk_path = get_local_disk_path()
    assert local_disk_path is not None

    # Find the scratch folder with the most free space.
    scratch_path = max((shutil.disk_usage(dir_path).free, dir_path)
                       for dir_path in local_disk_path.iterdir()
                       if dir_path.stem.startswith('scr'))[1]

    dst_path = scratch_path / dst_sub_path
    if not dst_path.parent.exists():
        dst_path.parent.mkdir(parents=True)

    return shutil.copy(src_path, dst_path)
