from extrapolation.datasets import utils
from extrapolation.data_utils import get_split_idxs
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import geopandas as gpd


# Paths to differents folders/files.
DATA_HOME = Path('/u/nlp/data')
CROPLAND_HOME = DATA_HOME / 'crops'
PATCH_DIR_PATH = CROPLAND_HOME / 'patches_2017'
CONDENSED_PATCH_DIR_PATH = CROPLAND_HOME / 'patches_2017_condensed'
LABEL_DIR_PATH = CROPLAND_HOME / 'patches_labels_crop_2017'
FULL_DATA_CACHE = CROPLAND_HOME / 'dataset.pkl'
CONDENSED_DATA_CACHE = CROPLAND_HOME / 'dataset_condensed.pkl'
SHAPEFILE_PATH = CROPLAND_HOME / 'shapefiles/cb_2018_us_state_500k.shp'

# Geographic layout of patches.
PATCH_SHAPE = (50, 50, 18)
NUM_PATCHES_PER_ROW = 592
NUM_PATCHES_PER_COL = 328
NUM_RECTS_PER_ROW = 8
NUM_RECTS_PER_COL = 8
NUM_PATCHES_PER_RECT_ROW = NUM_PATCHES_PER_ROW // NUM_RECTS_PER_ROW
NUM_PATCHES_PER_RECT_COL = NUM_PATCHES_PER_COL // NUM_RECTS_PER_COL
NUM_PATCHES_PER_RECT = NUM_PATCHES_PER_RECT_ROW * NUM_PATCHES_PER_RECT_COL
TOP_LAT = 41.5
BOTTOM_LAT = 37
LEFT_LON = -94
RIGHT_LON = -86
LAT_PER_PATCH = (TOP_LAT - BOTTOM_LAT) / NUM_PATCHES_PER_COL
LON_PER_PATCH = (RIGHT_LON - LEFT_LON) / NUM_PATCHES_PER_ROW


STATE_NAMES = ('Iowa', 'Missouri', 'Indiana', 'Illinois', 'Kentucky')
STATE_TO_INDEX = {state: index for index, state in enumerate(STATE_NAMES)}
VALID_SPLITS = ('train', 'val', 'test', 'test2')


def create_mmapped_file(patch_folder_path, mmap_path, overwrite=False):
    mmap_path = Path(mmap_path)
    if mmap_path.exists() and not overwrite:
        error_str = f'{str(mmap_path)} already exists!'
        error_str += ' Use overwrite=True if you want to overwrite.'
        raise ValueError(error_str)
    patch_folder_path = Path(patch_folder_path)
    num_patches = sum(1 for _ in patch_folder_path.iterdir())
    shape = (num_patches, *PATCH_SHAPE)
    mmap_array = np.memmap(mmap_path, dtype='float64', mode='w+', shape=shape)
    for idx, patch_path in enumerate(sorted(patch_folder_path.iterdir())):
        if idx % 10000 == 0:
            print(f'Processing patch {idx + 1}/{num_patches}')
        patch_array = np.load(str(patch_path), mmap_mode='r')
        mmap_array[idx] = patch_array
        del patch_array
    del mmap_array  # Flushes contents to file.


def resave_dataset(full_dataset_path, new_dataset_path, bands, dtype):
    '''
    Resaves the Landsat dataset into a new location with the specified bands
    and data type. Useful for condensing the dataset to have fewer bands and
    use a smaller numeric type (e.g., 32 bits instead of 64 bits).

    Parameters
    ----------
    full_dataset_path : Union[str, pathlib.Path]
        Path to the directory storing the full, original dataset.
    new_dataset_path : Union[str, pathlib.Path]
        Path to the directory that will hold the condensed dataset.
    bands : List[int]
        Specifies bands to keep in each patch. The bands range from 0 to 17.
    dtype : str
        The data type for the NumPy array (e.g., 'float32').
    '''
    full_dataset_path = Path(full_dataset_path)
    new_dataset_path = Path(new_dataset_path)
    for patch_file_path in full_dataset_path.iterdir():
        new_patch_path = new_dataset_path / patch_file_path.name
        if new_patch_path.exists():  # We've seen this file, skip it.
            continue
        full_patch = np.load(patch_file_path)
        new_patch = full_patch[:, :, bands].astype(dtype)
        np.save(new_patch_path, new_patch)


def remap_patch_num(patch_num):
    '''
    Remaps a Landsat patch number to an index in row-major order (i.e., left to
    right, top to bottom). The patch numbers are originally in row-major
    relative to their surrounding rectangular regions (see Section 3.5 in
    "Weakly Supervised Deep Learning for Segmentation of Remote Sensing
    Imagery" by Wang et al. for more details). This function maps the patch
    numbers so that they are in row-major order with respect to the entire
    geographic region.

    Parameters
    ----------
    patch_num : int
        Patch number to remap.

    Returns
    -------
    int
        Remapped patch number.
    '''
    # Compute row and column of surrounding rectangle.
    rect_idx = patch_num // NUM_PATCHES_PER_RECT
    rect_row = rect_idx // NUM_RECTS_PER_ROW
    rect_col = rect_idx % NUM_RECTS_PER_ROW

    # Compute top-left index in rectangle.
    first_idx = rect_row * NUM_PATCHES_PER_RECT * NUM_RECTS_PER_ROW
    first_idx += rect_col * NUM_PATCHES_PER_RECT_ROW

    # Compute row and column of patch relative to rectangle.
    inner_idx = patch_num % NUM_PATCHES_PER_RECT
    inner_row = inner_idx // NUM_PATCHES_PER_RECT_ROW
    inner_col = inner_idx % NUM_PATCHES_PER_RECT_ROW

    # Returns remapped patch number.
    return first_idx + inner_row * NUM_PATCHES_PER_ROW + inner_col


def patch_num_to_lat_lon(patch_num):
    '''
    Converts a patch number into a lat/lon coordinate. Each patch is 50 x 50
    pixels, where each pixel is 30m x 30m. The entire region is 328 x 592 in
    terms of patches. The patch numbers are ordered from left to right, top to
    bottom.

    Parameters
    ----------
    patch_num : int
        Patch number.

    Returns
    -------
    lat_lon : Tuple[int, int]
        Latitude and longitude of the center of the patch.
    '''
    patch_num = remap_patch_num(patch_num)
    row_num = patch_num // NUM_PATCHES_PER_ROW
    col_num = patch_num % NUM_PATCHES_PER_ROW
    lat = TOP_LAT - (row_num + 0.5) * LAT_PER_PATCH
    lon = LEFT_LON + (col_num + 0.5) * LON_PER_PATCH
    return np.array([lat, lon])


def load_from_disk(patch_dir_path=CONDENSED_PATCH_DIR_PATH,
                   label_dir_path=LABEL_DIR_PATH, in_memory=True):
    '''
    Loads Cropland dataset stored in the given folders.

    Parameters
    ----------
    patch_dir_path : Union[str, pathlib.path], default None
        Path to folder containing Cropland patches.
    label_dir_path : Union[str, pathlib.Path], default LABEL_FOLDER_PATH
        Path to folder containing Cropland labels.
    in_memory : bool, default True
        Whether to store the patches in memory or instead read them from the
        filesystem during iteration through the dataset. Note that this latter
        method is much more time-consuming because of the I/O bottleneck.

    Returns
    -------
    Dict[str, numpy.ndarray]
        Dict of data, targets, patch numbers.
    '''
    patch_dir_path = Path(patch_dir_path)
    label_dir_path = Path(label_dir_path)
    dataset_size = sum(1 for _ in label_dir_path.iterdir())
    data = []
    labels = np.empty(dataset_size, dtype=bool)
    patch_nums = np.empty(dataset_size, dtype=int)
    lat_lons = np.empty((dataset_size, 2))
    for i, label_file_path in enumerate(sorted(label_dir_path.iterdir())):
        labels[i] = np.load(label_file_path)[0]
        patch_num = int(label_file_path.stem[len('patch'):-len('_label_crop')])
        patch_nums[i] = patch_num
        lat_lon = patch_num_to_lat_lon(patch_num)
        lat_lons[i] = np.array(lat_lon)
        if in_memory:
            patch_path = patch_dir_path / f'patch{patch_num}.npy'
            if not patch_path.exists():  # Try loading condensed version.
                patch_path = patch_dir_path / f'patch{patch_num}_condensed.npy'
            data.append(np.load(patch_path))

    data_map = {}
    data_map['data'] = np.array(data)
    data_map['targets'] = labels
    data_map['patch_nums'] = patch_nums
    data_map['lat_lons'] = lat_lons
    return data_map


def load_data(patch_dir_path=CONDENSED_PATCH_DIR_PATH,
              label_dir_path=LABEL_DIR_PATH, in_memory=True, use_cache=True,
              try_local=True, copy_local=False, save_cache=False,
              cache_path=CONDENSED_DATA_CACHE):
    '''
    Reads the Cropland dataset from the filesystem and returns a map of the
    data. Can use a cached .pkl file for efficiency if desired.

    Parameters
    ----------
    patch_dir_path : Union[str, pathlib.Path], default CONDENSED_PATCH_DIR_PATH
        Path to the folder containing Cropland patches. If in_memory is False,
        then this parameter is reset to None to indicate the patches will not
    label_dir_path : Union[str, pathlib.Path], default LABEL_DIR_PATH
        Path to the folder containing Cropland labels.
    in_memory : bool, default True
        Whether to store the patches in memory or instead read them from the
        filesystem during iteration through the dataset. Note that this latter
        method is much more time-consuming because of the I/O bottleneck.
    use_cache : bool, default True
        Whether to use a cached form of the dataset. The cache_path
        parameter must also be present.
    try_local : bool, default True
        Whether to try searching for the cache file on the current machine's
        local disk. Useful since loading is often faster from the local disk
        rather than the distributed file system. Only relevant if use_cache is
        True.
    copy_local : bool, default False
        Whether to copy the cache file from the distributed file system to the
        local disk if the cache file is not already present. Useful for future
        runs that will use the same machine. Only relevant if use_cache and
        try_local are True.
    save_cache : bool, default False
        Whether to save the loaded data as a .pkl file for future use. The
        cache_path parameter must also be present.
    cache_path : Union[str, pathlib.Path], default CONDENSED_DATA_CACHE
        Path to .pkl file for loading/saving the Cropland dataset.

    Returns
    -------
    Dict[str, Union[numpy.ndarray, str]]
        Tuple of (labels, patch numbers).
    '''
    data_map = {}
    if use_cache:  # Default use cache.
        if try_local:
            local_cache_path = utils.search_local(cache_path)
            if local_cache_path is None:
                if copy_local:
                    src_path = DATA_HOME / cache_path
                    cache_path = utils.copy_to_local(src_path, cache_path)
                else:
                    cache_path = DATA_HOME / cache_path
            else:
                cache_path = local_cache_path
        with open(cache_path, 'rb') as pkl_file:
            data_map = pickle.load(pkl_file)
    else:
        data_map = load_from_disk(patch_dir_path, label_dir_path, in_memory)
        data_map['patch_dir_path'] = Path(patch_dir_path)

    if save_cache:  # Default don't save.
        with open(cache_path, 'wb') as pkl_file:
            pickle.dump(data_map, pkl_file, protocol=4)

    return data_map


def split_by_lat(data_map, lat_split):
    '''
    Filters data based on a certain latitude threshold.

    Parameters
    ----------
    data_map : dict
        Stores Landsat measurements, lat/lons, targets, and patch numbers.
    lat_split : Callable[[float], bool]
        Returns whether a point should be kept based on its latitude.
    '''
    data_map = data_map.copy()  # Don't overwrite original data_map.
    lat_mask = lat_split(data_map['lat_lons'][:, 0])
    for key in data_map.keys():
        if isinstance(data_map[key], np.ndarray) and data_map[key].size > 0:
            data_map[key] = data_map[key][lat_mask]
    return data_map


def split_by_lon(data_map, lon_split):
    '''
    Filters data based on a certain longitude threshold.

    Parameters
    ----------
    data_map : Dict[str, Union[numpy.ndarray, str]
        Stores Landsat measurements, lat/lons, targets, and patch numbers.
    lon_split : Callable[[float], bool]
        Returns whether a point should be kept based on its longitude.
    '''
    data_map = data_map.copy()  # Don't overwrite original data_map.
    lon_mask = lon_split(data_map['lat_lons'][:, 1])
    for key in data_map.keys():
        if isinstance(data_map[key], np.ndarray) and data_map[key].size > 0:
            data_map[key] = data_map[key][lon_mask]
    return data_map


def state_split(data_map, state_name, shapefile_path=SHAPEFILE_PATH):
    '''
    Filters data to only include points in the specified state. Note that
    pygeos must be installed for the geopandas.sjoin call to work.

    Parameters
    ----------
    data_map : Dict[str, Union[str, numpy.ndarray]]
        Stores Landsat measurements, lat/lons, and patch numbers.
    state_name : str
        Specifies U.S. state to filter on.
    shapefile_path : Union[str, pathlib.Path], default SHAPEFILE_PATH
        Path to .shp file that stores state boundaries.

    Returns
    -------
    Dict[str, Union[str, numpy.ndarray]]
        Shallow copy of data_map whose points have been filtered.
    '''
    state_name = state_name.capitalize()
    state_bounds_gdf = gpd.read_file(shapefile_path)
    lats = data_map['lat_lons'][:, 0]
    lons = data_map['lat_lons'][:, 1]
    geometry = gpd.points_from_xy(lons, lats)
    data_gdf = gpd.GeoDataFrame(crs=state_bounds_gdf.crs, geometry=geometry)
    state_boundary_gdf = state_bounds_gdf[state_bounds_gdf.NAME == state_name]
    state_points_gdf = gpd.sjoin(data_gdf, state_boundary_gdf)
    indices = state_points_gdf.index.to_numpy()
    return indices


def assign_states(data_map, state_to_index=STATE_TO_INDEX,
                  shapefile_path=SHAPEFILE_PATH):
    data_map['states'] = -np.ones(len(data_map['data']), dtype=int)
    lats = data_map['lat_lons'][:, 0]
    lons = data_map['lat_lons'][:, 1]
    geometry = gpd.points_from_xy(lons, lats)
    state_bounds_gdf = gpd.read_file(shapefile_path)
    data_gdf = gpd.GeoDataFrame(crs=state_bounds_gdf.crs, geometry=geometry)
    for state in state_to_index.keys():
        state_boundary_gdf = state_bounds_gdf[state_bounds_gdf.NAME == state]
        state_points_gdf = gpd.sjoin(data_gdf, state_boundary_gdf)
        indices = state_points_gdf.index.to_numpy()
        data_map['states'][indices] = state_to_index[state]
    assert np.all(data_map['states'] != -1)  # Ensure every point has a state.


def state_OOD_split(data_map, state_name, shapefile_path=SHAPEFILE_PATH):
    '''
    Takes an existing dictionary storing Landsat data and splits out the data
    for the desired state (e.g., to create an OOD split where the given state
    is OOD and the remaining states are in-domain).

    Parameters
    ----------
    data_map : Dict[str, Union[str, numpy.ndarray]]
        Dictionary storing Landsat measurements.
    state_name : str
        Specifies U.S. state to separate.
    shapefile_path : Union[str, pathlib.path], default SHAPEFILE_PATH
        Path to .shp file storing U.S. state boundaries.

    Returns
    -------
    Tuple whose first element is remaining states' data map and second element
    is the separated state's data map.
    '''
    indices_OOD = state_split(data_map, state_name, shapefile_path)
    all_indices = np.arange(len(data_map['data']))
    indices_ID = np.setdiff1d(all_indices, indices_OOD, assume_unique=True)
    return indices_ID, indices_OOD


def minnesota_split(data_map):
    data_map = split_by_lat(data_map, lambda lat: lat >= 40.5)
    data_map = split_by_lon(data_map, lambda lon: lon <= -91)
    return data_map


def illinois_split(data_map):
    data_map = split_by_lat(data_map, lambda lat: lat >= 38)
    data_map = split_by_lon(data_map, lambda lon: lon >= -90)
    data_map = split_by_lon(data_map, lambda lon: lon <= -87.5)
    return data_map


def indiana_split(data_map):
    data_map = split_by_lat(data_map, lambda lat: lat >= 37.5)
    data_map = split_by_lon(data_map, lambda lon: lon > -87.5)
    return data_map


def iowa_split(data_map):
    data_map = split_by_lat(data_map, lambda lat: lat < 40.5)
    data_map = split_by_lon(data_map, lambda lon: lon <= -91.5)
    return data_map


def kentucky_split(data_map):
    data_map = split_by_lat(data_map, lambda lat: lat < 37.5)
    data_map = split_by_lon(data_map, lambda lon: lon >= -88)
    return data_map


def shuffle_data(data_map, seed):
    '''
    Shuffles the data according to the given random seed.

    Parameters
    ----------
    data_map : Dict[str, Union[str, numpy.ndarray]]
        Stores Landsat labels and patch numbers.
    seed : int
        Seed for the NumPy random number generator.
    '''
    if not isinstance(seed, int):
        raise ValueError('Must pass an int for "seed" parameter!')

    rng = np.random.default_rng(seed)
    data_map = data_map.copy()
    permutation = rng.permutation(data_map['targets'].size)
    return utils.filter_data_map(data_map, permutation)


def normalize_lat_lon(lat_lon):
    '''
    Normalizes lat/lon coordinates into a unit square centered at the origin.

    Parameters
    ----------
    lat_lon : numpy.ndarray
        NumPy array of size 2 storing the latitude and longitude.

    Returns
    -------
    numpy.ndarray
        NumPy array of size 3 storing the (x, y, z) coordinates.
    '''
    lat, lon = lat_lon
    lat = (lat - BOTTOM_LAT) / (TOP_LAT - BOTTOM_LAT)
    lon = (lon - LEFT_LON) / (RIGHT_LON - LEFT_LON)
    lat_lon = np.array([lat, lon])
    return (lat_lon - 0.5) * 2


def split_unlabeled(all_indices, unlabeled_prop, seed):
    '''
    Holds out a specified proportion of data as unlabeled.

    Parameters
    ----------
    data_map : Dict[str, numpy.ndarray]
        Stores Landsat measurements, lat/lons, labels, and patch numbers.
    unlabeled_prop : float
        Proportion of data to hold out as unlabeled.
    seed : float
        Seed for random number generator for random selection of data.

    Returns
    -------
    Tuple[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]]
        Tuple of data dictionaries. The first dictionary contains the labeled
        data, and the second element contains the unlabeled data.
    '''
    num_unlabeled_points = int(all_indices.size * unlabeled_prop)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(all_indices)
    unlabeled_indices = permutation[:num_unlabeled_points]
    labeled_indices = permutation[num_unlabeled_points:]
    return labeled_indices, unlabeled_indices


def subsample_dataset(data_map, subsample_size, seed):
    '''
    Randomly chooses the specified number of points from the dataset and
    returns their indices.

    Parameters
    ----------
    data_map : Dict[str, numpy.ndarray]
        Stores Cropland dataset.
    subsample_size : int
        The size of the sample. If negative, then the entire dataset is taken.
    seed : int
        Seed for the random number generator that takes the sample.

    Returns
    -------
    numpy.ndarray[int]
        Indices of the subsampled dataset.
    '''
    if subsample_size < 0:
        subsample_size = len(data_map['data'])
    subsample_size = min(subsample_size, len(data_map['data']))
    rng = np.random.default_rng(seed)
    return rng.choice(len(data_map['data']), subsample_size, replace=False)


def create_lat_lon_patches(patch, lat_lon, normalize=False):
    num_pixels = np.prod(patch.shape[1:])
    if normalize:
        lat_lon = normalize_lat_lon(lat_lon)
    lat_lon_bands = np.repeat(lat_lon, (num_pixels,) * lat_lon.size)
    lat_lon_bands = lat_lon_bands.reshape((lat_lon.size, *patch.shape[1:]))
    if isinstance(patch, torch.Tensor):
        lat_lon_bands = torch.from_numpy(lat_lon_bands).type(patch.type())
    return lat_lon_bands


def append_lat_lon(patch, lat_lon, normalize=False):
    lat_lon_bands = create_lat_lon_patches(patch, lat_lon, normalize)
    if isinstance(patch, torch.Tensor):
        return torch.cat((patch, lat_lon_bands))
    else:
        return np.concatenate((patch, lat_lon_bands))


class Cropland(Dataset):
    '''
    PyTorch Dataset class around the Cropland dataset.
    '''
    def __init__(self, split='train', eval_mode=False, use_template=False,
                 template_dataset=None, in_bands=[10, 11], out_bands=None,
                 transform=None, target_transform=None, subsample_size=-1,
                 dataset_limit=-1, shuffle=False, shuffle_domains=False,
                 seed=None, unlabeled_prop=0, use_unlabeled=False,
                 use_unlabeled_id=False, use_unlabeled_ood=False,
                 masked_pretrain=False, include_lat_lon=False,
                 equal_unlabeled=False, target_lat_lon=False,
                 unlabeled_targets_path=None, **kwargs):
        '''
        Reads, shuffles, and splits data.

        Parameters
        ----------
        split : str, default 'train'
            Describes how to split dataset (e.g., 'train', 'val', 'test'). If
            this string starts with 'north' or 'south', then splitting by
            hemisphere occurs.
        eval_mode : bool, default False
            Whether this a dataset for evaluation.
        in_bands : List[int], default [10, 11]
            The indices of bands of the Landsat data to use as inputs. The
            default indices refer to the GCVI and NDVI bands.
        out_bands : List[int], default None
            Bands to use as output, likely for a pre-training task. If None,
            then the output is instead assumed to be the patch-level binary
            label for cropland/not cropland.
        transform : torch.nn.Module, default None
            Input transformation.
        target_transform : torch.nn.Module, default None
            Target transformation.
        template_dataset : torch.utils.data.Dataset, default None
            Dataset to help with initialization (for example, to use the same
            underlying NumPy arrays or standardization mean/stdevs).
        dataset_limit : int, default -1
            Caps the size of the dataset. If a negative number is passed, then
            the size is not limited.
        shuffle : bool, default False
            Whether to shuffle the entire dataset. A valid seed must be passed.
        shuffle_domains : bool, default False
            Whether to shuffle the order of domains before splitting. A valid
            seed must be passed.
        seed : int, default None
            Random seed to use for consistent shuffling.
        unlabeled_prop : float, default 0
            How much data from the entire dataset to keep as unlabeled data.
        use_unlabeled_id : bool, default False
            Whether to use the in-domain unlabeled data in training.
        use_unlabeled_ood : bool, default False
            Whether to use the out-of-domain unlabeled data in training.
        equal_unlabeled : bool, default False
            Whether to equalize sizes of ID and OOD unlabeled data.
        **kwargs:
            Passed through to load_data() function.
        '''
        assert '-' in split, 'Split must be of form {ood split}-{split type}'
        ood_split, split_type = split.split('-')
        if split_type not in VALID_SPLITS:
            raise ValueError('Invalid "split" parameter: {}'.format(split))

        self.eval_mode = eval_mode
        self.in_bands = in_bands
        self.out_bands = out_bands
        self.transform = transform
        self.target_transform = target_transform
        self.masked_pretrain = masked_pretrain
        self.include_lat_lon = include_lat_lon
        self.target_lat_lon = target_lat_lon

        use_unlabeled_all = use_unlabeled_id and use_unlabeled_ood
        use_unlabeled_all |= use_unlabeled
        self.use_unlabeled = use_unlabeled_id or use_unlabeled_ood
        self.use_unlabeled |= use_unlabeled

        if use_template and template_dataset is not None:
            self.data_map = template_dataset.data_map
        else:
            self.data_map = load_data(**kwargs)

        states = ood_split[3:] if ood_split.startswith('non') else ood_split
        states = states.split('_')
        states = [state.lower().capitalize() for state in states]
        state_indices = [STATE_TO_INDEX[state] for state in states]
        ood_mask = np.isin(self.data_map['states'], state_indices, invert=True)
        if ood_split.startswith('non') ^ (split_type == 'test2'):
            ood_mask = ~ood_mask

        split_idxs = get_split_idxs(unlabeled_prop, np.where(ood_mask)[0],
                                    len(self.data_map['data']), seed)

        sub_indices = subsample_dataset(self.data_map, subsample_size, seed)
        labeled_indices = np.intersect1d(split_idxs[split_type], sub_indices,
                                         assume_unique=True)
        id_size = split_idxs['unlabeled_id'].size
        ood_size = split_idxs['unlabeled_ood'].size
        if equal_unlabeled:
            min_size = min(id_size, ood_size)
            min_size -= min_size % 2  # Ensure min_size is even.
            id_size = min_size
            ood_size = min_size

        unlabeled_id = np.empty(0, dtype=int)
        unlabeled_ood = np.empty(0, dtype=int)
        if use_unlabeled_all:
            if equal_unlabeled:
                id_size = id_size // 2
                ood_size = ood_size // 2
            use_unlabeled_id = True
            use_unlabeled_ood = True
        if use_unlabeled_id:
            unlabeled_id = split_idxs['unlabeled_id']
            #rng = np.random.default_rng(seed + 1)
            #unlabeled_id = rng.choice(unlabeled_id, id_size, replace=False)'''
        if use_unlabeled_ood:
            unlabeled_ood = split_idxs['unlabeled_ood']
            #rng = np.random.default_rng(seed + 2)
            #unlabeled_ood = rng.choice(unlabeled_ood, ood_size, replace=False)

        unlabeled_indices = np.hstack((unlabeled_id, unlabeled_ood))
        if shuffle:
            #rng = np.random.default_rng(seed + 3)
            rng = np.random.default_rng(seed)
            rng.shuffle(unlabeled_indices)
            rng.shuffle(labeled_indices)

        targets = self.data_map['targets']
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self._unseen_unlabeled_targets = targets[self.unlabeled_indices]

        # assumes that unlabeled_targets_path uses the same split
        self.unlabeled_targets_path = unlabeled_targets_path
        if unlabeled_targets_path is not None and not self.eval_mode:
            with open(unlabeled_targets_path, 'rb') as pkl_file:
                pseudolabels_dict = pickle.load(pkl_file)
            pseudolabels = pseudolabels_dict['pseudolabels']
            assert len(pseudolabels) == len(self.unlabeled_indices)
            acc = np.mean(pseudolabels == targets[self.unlabeled_indices])
            assert acc == pseudolabels_dict['pseudolabel_acc']
            self.data_map['targets'][self.unlabeled_indices] = pseudolabels

    def __getitem__(self, index):
        '''
        Returns a dictionary of elements from the Dataset for the given index.

        Parameters
        ----------
        index : int
            Index into the Dataset.

        Returns
        -------
        Dict[str, Union[numpy.ndarray, torch.Tensor, int]]
            Dictionary with keys 'data', 'target', and 'domain_label'.
        '''
        if index < len(self.labeled_indices):
            index = self.labeled_indices[index]
            labeled = True
        elif self.use_unlabeled and index < len(self):
            index -= len(self.labeled_indices)
            index = self.unlabeled_indices[index]
            labeled = False
        else:
            raise IndexError('Dataset index out of range.')

        img = self.data_map['data'][index]
        img = np.moveaxis(img, source=-1, destination=0)
        lat_lon = self.data_map['lat_lons'][index]
        patch_num = self.data_map['patch_nums'][index]
        state = self.data_map['states'][index]
        metadata = {'patch_num': patch_num, 'lat_lon': lat_lon, 'state': state,
                    'labeled': labeled}
        if self.masked_pretrain:
            input_as_target = img[self.in_bands, 1:-1, 1:-1]
            output_as_target = img[self.out_bands, 1:-1, 1:-1]
            if self.target_transform is not None:
                input_as_target = self.target_transform(input_as_target)
                output_as_target = self.target_transform(output_as_target)
            target = [-100, input_as_target, output_as_target]
            img = img[self.in_bands + self.out_bands, ...]
            if self.transform is not None:
                img = self.transform(img)

            use_idx = np.random.choice(a=[1, 2])
            metadata['use_idx'] = use_idx
            if use_idx == 1:
                img[:len(self.in_bands), :, :] = 0
            elif use_idx == 2:
                img[len(self.in_bands):, :, :] = 0
        else:
            if self.out_bands is not None:
                target = img[self.out_bands, 1:-1, 1:-1]
            else:
                bool_target = self.data_map['targets'][index]
                target = np.array([bool_target], dtype='float32')

            img = img[self.in_bands, ...]
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.transform is not None:
                img = self.transform(img)

            if self.include_lat_lon:
                img = append_lat_lon(img, lat_lon, normalize=True)

            if self.target_lat_lon:
                assert self.out_bands is not None
                target = append_lat_lon(target, lat_lon, normalize=True)

        return {'data': img, 'target': target, 'domain_label': metadata}

    def __len__(self):
        length = len(self.labeled_indices)
        if self.use_unlabeled and not self.eval_mode:
            length += len(self.unlabeled_indices)
        return length

    def get_mean(self):
        '''
        Returns the mean for this dataset. Useful for getting the mean of a
        training set in order to standardize val and test sets.

        Returns
        -------
        self.mean, which is a float or numpy.ndarray.
        '''
        return self.mean

    def set_mean(self, mean):
        '''
        Sets the mean to use for standardization. Useful for setting the mean
        of a val or test set from the mean of a training set.

        Parameters
        ----------
        mean : Union[float, numpy.ndarray]
            Mean to subtract from data.
        '''
        self.mean = mean

    def get_std(self):
        '''
        Returns the std for this dataset. Useful for getting the std of a
        training set in order to standardize val and test sets.

        Returns
        -------
        self.std, which is a float or numpy.ndarray.
        '''
        return self.std

    def set_std(self, std):
        '''
        Sets the std to use for standardization. Useful for setting the std of
        a val or test set from the std of a training set.

        Parameters
        ----------
        std : Union[float, numpy.ndarray]
            Std to divide by for standardization.
        '''
        self.std = std
