#!/usr/bin/bash

DIRECTORY=${1:-"./data/training_data/perqa"}

# load conda
eval "$(conda shell.bash hook)"
conda activate reqap

# train set
OUTPUT_PATH="$DIRECTORY/crossencoder/train_data.jsonl"
for i in {0..11}
do 
	PERSONA="train_persona_${i}"
	OUT="$DIRECTORY/crossencoder/personas/$PERSONA.jsonl"
    [ -f "$OUT" ] && cat "$OUT" >> "$OUTPUT_PATH"
done

# dev set
OUTPUT_PATH="$DIRECTORY/crossencoder/dev_data.jsonl"
for i in {0..1}
do 
	PERSONA="dev_persona_${i}"
    OUT="$DIRECTORY/crossencoder/personas/$PERSONA.jsonl"
    [ -f "$OUT" ] && cat "$OUT" >> "$OUTPUT_PATH"
done

echo "Done"
