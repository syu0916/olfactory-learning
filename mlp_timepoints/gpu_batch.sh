#!/bin/bash
#SBATCH --nodes=1              # node count
#SBATCH -p gpu                 # partition (queue)
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --ntasks-per-node=1    # total number of tasks across all nodes
#SBATCH --cpus-per-task=2      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-gpu=99G      # memory per GPU
#SBATCH -t 16:00:00            # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=sean_yu@brown.edu

module purge
unset LD_LIBRARY_PATH
echo here
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"

# srun apptainer exec --nv ../tensorflow-24.03-tf2-py3.simg bash -c "python3 main.py"
# srun apptainer exec --nv ../tensorflow-24.03-tf2-py3.simg bash -c "python3 main.py"

srun apptainer exec --nv ../tensorflow-24.03-tf2-py3.simg bash -c "python3 shap_script2.py"
# srun apptainer exec --nv ../tensorflow-24.03-tf2-py3.simg python3 interpretation.py
# srun apptainer exec --nv ../tensorflow-24.03-tf2-py3.simg python3 preprocess.py