# XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics
## Introduction
We construct a quality-parallel dataset by injecting varying numbers of Multidimensional Quality Metric [(MQM)](https://github.com/google/wmt-mqm-human-evaluation?tab=readme-ov-file)-defined errors into high-quality translations, enabling controlled and comparable translation quality across languages. This dataset serves as a benchmark for the systematic evaluation of cross-lingual scoring bias in evaluation metrics.
The figure illustrates our dataset construction pipeline, which is highly flexible and can be readily adapted or extended to different languages and error types.

<img width="1139" height="361" alt="pipeline" src="https://github.com/user-attachments/assets/4f27314e-d038-45b6-ba5d-2f12be12e2d5" />






## Benchmark
- High Quality Translation Dataset: [Flores+](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- Language Pairs: English-Chinese (en-zh); English-Lao (en-lo); English-Japanese (en-ja); English-Spanish (en-es); English-French (en-fr); English-Indonesian (en-id); English-Vietnamese (en-vi); English-German (en-de); English-Sinhala (en-si).
- Error Types: Addition; Omission; Mistranslation; Untranslated.
- Triplet Count Distribution (quality level represents the number of errors present in translations):

| Quality Level | en-zh | en-lo | en-ja | en-vi | en-id | en-fr | en-es | en-si | en-de |
|:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 1 | 776 | 753 | 775 | 771 | 782 | 775 | 771 | 765 | 774 |
| 2 | 2,109 | 2,053 | 2,078 | 2,056 | 2,095 | 1,992 | 2,016 | 2,064 | 2,049 |
| 3 | 2,548 | 2,627 | 2,441 | 2,420 | 2,421 | 2,068 | 2,233 | 2,489 | 2,337 |
| 4 | 1,466 | 1,704 | 1,324 | 1,387 | 1,311 | 957 | 1,069 | 1,432 | 1,234 |
| 5 | 406 | 558 | 340 | 428 | 312 | 198 | 203 | 361 | 313 |

## Installation
```bash
srun -p gpu_intr --gres=gpu:6000:1 --pty bash

conda create -n metric python=3.10
conda activate metric

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers evaluate
pip install unbabel-comet

# when using comet, please ensure you acknowledge the license on each page
# https://huggingface.co/Unbabel/wmt22-cometkiwi-da
# https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl
# https://huggingface.co/Unbabel/XCOMET-XL
# please run huggingface-cli login
# input your tokens

git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
python3 -m zipfile -e BLEURT-20.zip .
rm BLEURT-20.zip
cd ..

git clone https://github.com/google-research/metricx.git
pip install jsonlines
pip install accelerate
pip install tf-keras
```

## Repository Structure
```text
scale/
├── functions/                 
│   ├── compute_scores.py      # Compute metric scores
│   ├── prompt_tools.py        # Prompt construction for GPT-4o
│   ├── score_tools.py         # Scoring utilities
│   ├── sample.py              # Sample triplets across different quality levels
│   └── table_gather.py        # Aggregate score results into tables
│
├── results/                   # GPT-4o outputs with single injected errors (4 types)
├── merged_result/             # Outputs with multiple errors formed by merging single-error outputs
│
├── call_gpt.py                # Instruct GPT-4o to inject errors into translations
├── compute_scores.sh          # Script for score computation
├── merge.py                   # Construct multi-error translations by merging single-error outputs
                 
```
## Evalution
You can obtain metric scores (spBLEU; chrF++; BLEURT-20; COMET-22; xCOMET-XL; MetricX-23; COMET-KIWI-22; COMET-KIWI-23; MetricX-23-QE. )by running:
```bash
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
```
## Citation
If you use XQ-MEval in your research, please cite our paper:
