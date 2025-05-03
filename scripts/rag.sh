#!/usr/bin/bash

FUNCTION=$1
CONFIG=${2:-"config/perqa/rag_openai.yml"}
SPLIT=${3:-""}
PERSONA=${4:-""}

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
OUT="logs/$BENCHMARK/rag$FUNCTION$SPLIT-$CFG_NAME$PERSONA.log"

# start script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup
	export FUNCTION CONFIG OUT 
    nohup sh -c 'python -u rag.py ${FUNCTION} ${CONFIG}' > $OUT 2>&1 &
else
	# 
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$OUT
#SBATCH -o $OUT
#SBATCH -p $GPU
#SBATCH --gres gpu:$GPU_NUM
#SBATCH -t 0-02:00:00
#SBATCH -d singleton

python -u rag.py $FUNCTION $CONFIG $SPLIT $PERSONA
EOT
fi