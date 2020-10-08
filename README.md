# In-N-Out (repo in progress)

Main scripts: See `scripts/*_runner.py`, `scripts/run_main.sh`, `extrapolation/main.py`.

The main script is in `extrapolation/main.py`. The script requires a YAML config
file - an example is `extrapolation/configs/pacs/resnet18.yaml`.
To dynamically change values of the config file with command line arguments,
simply add new arguments of the form `--key=val` where the key can be any
string of multiple keys separated by periods. This is to allow for changing
nested components of the config file. For example `--model.args.depth=3` changes
the config dictionary in this way: `config['model']['args']['depth'] = 3`.
It is important that the key and value are separated by an equals sign.

We use weights and biases to plot graphs and figures while experiments are running.
We have a `p-lambda` team in Weights and Biases that should give us more W&B resources to use, which you can be added to.

## Steps to run an experiment

The first time you run this project, in the current directory, which contains README, create a virtualenv:
```
python3 -m venv .env
. .env/bin/activate
pip install -e .
```

In subsequent runs you typically only need to activate the environment (although you may need to sometimes upgrade packages, you can run the install command above after activate):
```
. .env/bin/activate
```

Cd into the extrapolation subfolder, which contains `main.py`. Then login and start weights and biases:
```
cd extrapolation
wandb login
wandb on
```

Finally, you can run one of the existing configs. Each config defines an experiment, and you can overwrite parameters.
Here's an example run which overwrites the optimizer learning rate to 0.001:
```
../.env/bin/python main.py --config=configs/synthetic/erm.yaml --model_dir=models/example_model --project_name=extrapolation --run_name=example_run --group_name=example_group --optimizer.args.lr=0.001
```
