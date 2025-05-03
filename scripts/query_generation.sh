#!/usr/bin/bash

FUNCTION=$1
CONFIG=${2:-"config/perqa/query_generation_openai.yml"}

GPU="gpu24"
GPU_NUM="1"

# derive config name
IFS='/' read -ra NAME <<< "$CONFIG"
LENGTH=$(( ${#NAME[@]} - 1 ))
CFG_NAME="${NAME[$LENGTH]%".yml"}"
BENCHMARK="${NAME[1]}"

# load conda
eval "$(conda shell.bash hook)"
conda activate reqap

# set log level
export LOGLEVEL="DEBUG"
export GPU_NUM
export TOKENIZERS_PARALLELISM="false"

# set output path
mkdir -p logs/$BENCHMARK
OUT="logs/$BENCHMARK/query_generation$FUNCTION-$CFG_NAME.log"

# start script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup
	export FUNCTION CONFIG OUT 
    nohup sh -c 'python -u query_generation.py ${FUNCTION} ${CONFIG}' > $OUT 2>&1 &
else
	# 
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$OUT
#SBATCH -o $OUT
#SBATCH -p $GPU
#SBATCH --gres gpu:$GPU_NUM
#SBATCH -t 0-03:00:00
#SBATCH -d singleton

python -u query_generation.py $FUNCTION $CONFIG
EOT
fi