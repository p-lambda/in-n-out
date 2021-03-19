#!/bin/bash

# run these first
# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_cl --mode baseline --trial ${trial}
# done
# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_cl --mode aux-inputs --trial ${trial}
# done
# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_cl --mode aux-outputs --trial ${trial}
# done

for trial in 1 2 3 4 5
do
python landcover_era5_exps.py --no_wandb --use_cl --mode in-n-out --trial ${trial}
done
