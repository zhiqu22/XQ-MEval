import functions.score_tools as tool
import pandas as pd
from pathlib import Path


# read data
# src = tool.read_txt_strip(file_url) -> list[str]
# src = ["hello, 1", "hello, 2"]
# hyp = ["hello, world 1", "hello, world 2"]
# ref = ["hello world 3", "hello world 4"]
# hyp_2 = ["hello, world 5", "hello, world 6"]
# ref_2 = ["hello world 7", "hello world 8"]
current_dir = Path.cwd()
file_path = current_dir/"scale" / "merged_result" / "en-fr-merge-1.tsv"
data = pd.read_csv(file_path, sep="\t")


# usage #1, call sequence metric for a single pair

# input_data = tool.prepare_input_data(None, hyp, ref, metric_type="seq", metric_name="chrf")
# results = tool.seq_score(input_data, metric_name="chrf")
# system_score, scores = results.system_score, results.scores
# print(results)

# usage #2, call bleurt for multiple pairs
# "multiple pairs" inlcude 1 pair, namely [hyp], [ref]
hyp = [[row["merged_mt"]] for _, row in data.iterrows()]
ref = [[row["ref"]] for _, row in data.iterrows()]

input_data, segments = tool.prepare_batch_bleurt_input_data(hyp,ref)
results = tool.bleurt_score(input_data, segments)
total_scores = 0
total_sum = 0
for i in range(len(hyp)):
    score = results[i].system_score 
    total_scores += 1  
    total_sum += score  
average_score = total_sum / total_scores if total_scores > 0 else 0
print(f"Average score: {average_score:.6f}")


# print(results)


# usage #3, single gpu mode in calling comet
# metric_type: ["qe", "reg"]
# metric_name: comet, xcomet, kiwi, kiwi23
# in this case, you have to assign device, e.g., cuda:0

# pair_1 = tool.prepare_input_data(src, hyp, ref, metric_type="qe", metric_name="kiwi")
# pair_2 = tool.prepare_input_data(src, hyp_2, ref_2, metric_type="qe", metric_name="kiwi")
# results = tool.comet_score([pair_1, pair_2], metric_name="kiwi", batch_size=2, device="cuda:0")
# print(results)


# usage #4, compared to usage #3, we can use multiple gpus
# do not assign device, but assign gpus = 2/4/8
# results = tool.comet_score([pair_1, pair_2], metric_name="kiwi", batch_size=4, gpus=1)
# print(results)

# usage #5, compute metricx with a single pair
# qe
# dir_path = tool.prepare_input_data(src, hyp, ref=None, metric_type="qe", metric_name="metricx")
# results = tool.metricx_score(dir_path, metric_type="qe", segments=None, batch_size=1)
# reg
# dir_path = tool.prepare_input_data(None, hyp, ref, metric_type="reg", metric_name="metricx")
# results = tool.metricx_score(dir_path, metric_type="reg", segments=None, batch_size=1)
# print(results)

# usage #6, compute metricx with multiple pairs
# the difference compared to #5 is that 'segments' is not None
# dir_path, segments = tool.prepare_batch_metricx_input_data(None, [hyp, hyp_2], [ref, ref_2], metric_type="reg")
# results = tool.metricx_score(dir_path, metric_type="reg", segments=segments, batch_size=1)
# print(results)
