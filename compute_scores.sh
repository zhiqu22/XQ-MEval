#!/bin/bash

cd work/scale
metrics=("spbleu" "chrf" "bleurt" "comet" "xcomet" "reg" "kiwi" "kiwi23" "qe")
tgt_langs=("de" "zh" "ja" "fr" "lo" "si" "id" "vi" "es")
gpus=8
batch_size=16
for metric in "${metrics[@]}"; do
    for tgt_lang in "${tgt_langs[@]}"; do
        python functions/compute_scores.py ${tgt_lang} ${metric} ${gpus} ${batch_size}
    done
done

python functions/table_gather.py

