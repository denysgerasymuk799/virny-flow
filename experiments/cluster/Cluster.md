# Cluster

## Useful Materials

NYU HPC dashboard -- [https://ood-4.hpc.nyu.edu/pun/sys/dashboard](https://ood-4.hpc.nyu.edu/pun/sys/dashboard)


## VirnyFlow Deployment

Start a VirnyFlow cluster in one SLURM job:
```shell
pip install -r requirements.txt

# Step 1: Create a sbatch file with all arguments, similar to virny-flow-experiments/cluster/cost_model_exp1/run-cost-model-exp1.sbatch
# Step 2: Execute the following command:
scratch/dh3553/projects/virny-flow-experiments/logs$ bash /home/dh3553/projects/virny-flow-experiments/cluster/scalability_exp/run-virny_flow-folk_emp_big.sh
```

Start Alpine Meadow and auto-sklearn cluster in one SLURM job:
```shell
home/dh3553/projects/virny-flow-experiments/logs$ bash ../cluster/scalability_exp/run-alpine_meadow-folk_emp_big.sh
```


## Development on the HPC cluster

SSH to the cluster:
```shell
ssh dh3553@greene.hpc.nyu.edu
```

Get your project IDs:
```shell
sacctmgr list assoc format=user,account%25 where user=<NYU id>
```

Commands for development on the cluster using GPUs:
```shell
# To request one GPU card, 16 GB memory, and 12 hour running duration
srun -t12:00:00 --mem=16000 --gres=gpu:rtx8000:1 --pty /bin/bash

singularity exec --nv --overlay /scratch/dh3553/virny_flow_project/vldb_sds_env.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

Singularity> source /ext3/env.sh
```

Commands for development on the cluster using CPUs:
```shell
# To request 4 CPUs, 16 GB memory, and 12 hour running duration
srun -t12:00:00 --mem=16000 --ntasks-per-node=1 --cpus-per-task=4 --pty /bin/bash

singularity exec --overlay /scratch/dh3553/virny_flow_project/vldb_sds_env.ext3:rw /scratch/work/public/singularity/ubuntu-22.04.1.sif /bin/bash

Singularity> source /ext3/env.sh
```

Run jobs using a bash script:
```shell
/logs $ bash ../cluster/run_baselines/run-baselines.sh
```

Uninstall a library and its dependencies:
```shell
pip install pip-autoremove
pip-autoremove library_name
```

Find location of a python package to change source files:
```shell
# /ext3/miniconda3/lib/python3.9/site-packages/datawig/__init__.py
python -c "import datawig; print(datawig.__file__)"
```

SSH to a bash SLURM job:
```shell
# Pattern: ssh username@hostname
ssh dh3553@cs223
```

Count the number of files in each directory:
```shell
du -a | cut -d/ -f2 | sort | uniq -c | sort -nr
```



## Singularity

Debugging:
```shell
singularity exec docker://bitnami/zookeeper:latest bash

singularity instance list

ps aux | grep zoo

tail -f /proc/1955710/fd/1
```



## Singularity Overlay Setup

Configurate dependencies on the cluster ([ref](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)):

```shell
cd /scratch/dh3553/virny_flow_project
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz

singularity exec --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash

wget https://repo.continuum.io/miniconda/Miniconda3-py39_24.9.2-0-Linux-x86_64.sh
bash Miniconda3-py39_24.9.2-0-Linux-x86_64.sh -b -p /ext3/miniconda3
# rm Miniconda3-py39_24.9.2-0-Linux-x86_64.sh # if you don't need this file any longer

# Next, create a wrapper script /ext3/env.sh using a text editor, like nano.
touch /ext3/env.sh
nano /ext3/env.sh

# The wrapper script will activate your conda environment, to which you will be installing your packages and dependencies.
# The script should contain the following:
"""
#!/bin/bash

unset -f which

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
"""

source /ext3/env.sh

# Install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
pip --version

pip install -r /home/dh3553/projects/virny-flow-experiments/requirements.txt

# https://stackoverflow.com/questions/54249577/importerror-libcuda-so-1-cannot-open-shared-object-file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/compat

exit

mv ./overlay-15GB-500K.ext3 ./vldb_sds_env.ext3
```
