#!/usr/bin/bash

CONFIG=${1:-"config/perqa/reqap_sft.yml"}

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
mkdir -p logs/$BENCHMARK/retrieval--derive-data-reqap_openai-parallel/

# train set
for i in {0..11}
do 
	PERSONA="train_persona_${i}"
	OUT="logs/$BENCHMARK/retrieval--derive-data-reqap_openai-parallel/$PERSONA.log"

	# run via nohup
	export FUNCTION CONFIG OUT PERSONA
	nohup sh -c 'python -u run_retrieval.py --derive_data ${CONFIG} ${PERSONA}' > $OUT 2>&1 &
done

# dev set
for i in {0..1}
do 
	PERSONA="dev_persona_${i}"
	OUT="logs/$BENCHMARK/retrieval--derive-data-reqap_openai-parallel/$PERSONA.log"

	# run via nohup
	export FUNCTION CONFIG OUT PERSONA
	nohup sh -c 'python -u run_retrieval.py --derive_data ${CONFIG} ${PERSONA}' > $OUT 2>&1 &
done