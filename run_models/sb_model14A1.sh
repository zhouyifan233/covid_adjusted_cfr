#!/bin/bash = 80% (A) (Bi et al)
#SBATCH --job-name='model14A1'
#SBATCH --partition=all
#SBATCH --array=1-4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=96:00:00

module load Boost/1.66.0-foss-2018a
./model14 sample num_warmup=500 num_samples=500 \
      adapt delta=0.8 algorithm=hmc engine=nuts max_depth=10 init=0.5 \
      data file=data_S_model14A1_2020-03-11-16-20-16.R output file=S_model14A1_2020-03-11-16-20-16_${SLURM_ARRAY_TASK_ID}.csv refresh=10
