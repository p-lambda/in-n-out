import subprocess
import shlex


def make_deps(exp_name):
    return [f"cropland_{exp_name}_trial{i}" for i in range(1, 6)]


def make_dirnames(exp_name):
    return [f"cropland_{exp_name}_trial{i}/models/cropland/cropland_{exp_name}_trial{i}" for i in range(1, 6)]


def make_dirnames_innout(exp_name, it=0):
    return [f"cropland_{exp_name}_trial{i}/models/cropland/cropland_{exp_name}_iter{it}_trial{i}" for i in range(1, 6)]


if __name__ == "__main__":
    exp_name = 'in-n-out'
    for it in [0, 1]:
        deps = make_deps(exp_name)
        deps = ' '.join([f':{dep}' for dep in deps])

        dirnames = ' '.join(make_dirnames_innout(exp_name, it=it))
        cmd = f'cl run -n cropland_{exp_name}_iter{it} -w in-n-out-iclr-cropland --request-docker-image ananya/in-n-out --request-queue tag=nlp {deps} :get_pastable.py "python get_pastable.py -a {dirnames}"'

        subprocess.run(shlex.split(cmd))

'''for exp_name in ['baseline', 'aux-inputs', 'aux-outputs']:
        deps = make_deps(exp_name)
        deps = ' '.join([f":{dep}" for dep in deps])

        dirnames = ' '.join(make_dirnames(exp_name))
        cmd = f'cl run -n cropland_{exp_name} -w in-n-out-iclr-cropland --request-docker-image ananya/in-n-out --request-queue tag=nlp {deps} :get_pastable.py "python get_pastable.py -a {dirnames}"'

        subprocess.run(shlex.split(cmd))'''
