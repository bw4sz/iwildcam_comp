#!/bin/bash
#SBATCH --job-name=DeepTrap   # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1            # Number of cores per MPI rank
#SBATCH --mem-per-cpu=40GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepTrap.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepTrap.err

ml git
ml tensorflow
export PATH=${PATH}:/home/b.weinstein/miniconda/envs/DeepTrap/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda/envs/DeepTrap/lib/python3.6/site-packages/
echo $PYTHONPATH

cd /home/b.weinstein/iwildcam_comp/DeepTrap/
python Locations.py
