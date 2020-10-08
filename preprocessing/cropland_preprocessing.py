from extrapolation.datasets import utils
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
