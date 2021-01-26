# In-N-Out (repo in progress)

Main scripts: See `scripts/*_runner.py`, `scripts/run_main.sh`, `extrapolation/main.py`.

The main script is in `innout/main.py`. The script requires a YAML config
file - an example is `innout/configs/pacs/resnet18.yaml`.
To dynamically change values of the config file with command line arguments,
simply add new arguments of the form `--key=val` where the key can be any
string of multiple keys separated by periods. This is to allow for changing
nested components of the config file. For example `--model.args.depth=3` changes
the config dictionary in this way: `config['model']['args']['depth'] = 3`.
It is important that the key and value are separated by an equals sign.

## Steps to run an experiment

The first time you run this project, in the current directory, which contains README, create a virtualenv:
```
python3 -m venv .env
. .env/bin/activate
pip install -e .
```

We used PyTorch 1.6.0 with Cuda 10.1 for our experiments. You can install this version via `pip` with the following command:

```
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

In subsequent runs you typically only need to activate the environment (although you may need to sometimes upgrade packages, you can run the install command above after activate):
```
. .env/bin/activate
```

Finally, you can run one of the existing configs. Each config defines an experiment, and you can overwrite parameters.
Here's an example run which overwrites the optimizer learning rate to 0.001:
```
../.env/bin/python main.py --config=configs/synthetic/erm.yaml --model_dir=models/example_model --project_name=innout --run_name=example_run --group_name=example_group --optimizer.args.lr=0.001
```
