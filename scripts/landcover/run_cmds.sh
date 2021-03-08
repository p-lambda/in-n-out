#!/bin/bash

# run these first
# for trial in 1 2 3 4 5
for trial in 5
do
python landcover_era5_exps.py --no_wandb --use_cl --mode baseline --trial ${trial}
done
# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_cl --mode aux-inputs --trial ${trial}
# done
# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_cl --mode aux-outputs --trial ${trial}
# done

# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_cl --mode in-n-out --trial ${trial} --dryrun
# done

# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_slurm --mode baseline --trial ${trial}
# python landcover_era5_exps.py --no_wandb --use_slurm --mode aux-inputs --trial ${trial}
# python landcover_era5_exps.py --no_wandb --use_slurm --mode aux-outputs --trial ${trial}
# done

# after these complete, run in-n-out
# for trial in 1 2 3 4 5
# do
# python landcover_era5_exps.py --no_wandb --use_slurm --mode in-n-out --trial ${trial}
# done

