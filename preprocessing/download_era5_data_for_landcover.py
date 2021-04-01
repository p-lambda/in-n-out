import math
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import cdsapi

import gdal


def download_era5():
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'format': 'grib',
            'variable': [
                '2m_temperature', 'mean_total_precipitation_rate', 'soil_type',
                'surface_net_solar_radiation', 'total_cloud_cover', 'total_precipitation',
            ],
            'product_type': 'monthly_averaged_reanalysis',
            'year': '2018',
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'time': '00:00',
        },
        args.save_pkl)



def open_grib(grib_file):
    ds = gdal.Open(grib_file, 0)
    gt = ds.GetGeoTransform()
    return ds, gt


def grib_data_nearest_point(ds, gt, lats, lons):
    """
    Gets data from grib grid that is closest to the queried lats-lons
    Adapted from https://gis.stackexchange.com/questions/325924/getting-data-closest-to-specified-point-in-a-grib-file-python
    """
    # get origin's coordinates, pixel width and pixel height
    # the GetGeoTransform method returns the skew in the x and y axis but you
    # can ignore these values
    ox, pw, xskew, oy, yskew, ph = gt

    vals_per_month = []

    # open the grib file, get the specified band and read it as an array
    for month in range(12):
        bands = [i + month * 6  for i in range(1, 7)]
        month_vals = []

        band_names = []
        for band_idx in bands:
            band = ds.GetRasterBand(band_idx)
            meta = band.GetMetadata()
            band_names.append(meta['GRIB_COMMENT'])
            arr = band.ReadAsArray()

            band_vals = []
            for lat, lon in zip(lats, lons):

                # calculate the indices (row and column)
                i = math.floor((oy - lat) / ph)
                j = math.floor((lon - ox) / pw)

                # index the array to return the correspondent value
                band_vals.append(arr[i, j])
            month_vals.append(band_vals)
        vals_per_month.append(month_vals)
    vals_per_month = np.asarray(vals_per_month).transpose((2, 1, 0))
    return vals_per_month, band_names


if __name__ == "__main__":
    import argparse
    parser.add_argument('--save_path', type=str, default='reanalysis_era5_single_levels_monthly_means.grib', help='where to save era5 data')
    parser.add_argument('--image_only_pkl', type=str, default='landcover_only_data.pkl', help='landcover data without era5')
    parser.add_argument('--output_pkl', type=str, default='landcover_data.pkl', help='landcover data with era5')
    args = parser.parse_args()

    download_era5()
    ds, gt = open_grib(args.save_path)

    with open(args.image_only_pkl, 'rb') as f:
        data_map = pickle.load(f)

    for k, v in tqdm(data_map.items()):
        lats = v[1][:, 0]
        lons = v[1][:, 1]

        vals_per_month, band_names = grib_data_nearest_point(ds, gt, lats, lons)
        new_tup = list(v)
        new_tup.append(vals_per_month)
        new_tup.append(band_names)
        data_map[k] = tuple(new_tup)

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(data_map, f)
