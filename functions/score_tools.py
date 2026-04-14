from typing import List
import sentencepiece as spm
import os, subprocess, shutil
from datetime import datetime
from sacrebleu.metrics import BLEU, CHRF
import jsonlines

class MyScores():
    def __init__(self, scores: List[float], system_score: float = None):
        self.scores = scores
        if system_score is None:
            self.system_score = sum(scores)/len(scores)
        else:
            self.system_score = system_score
    
    def __repr__(self):
        return f'{{"system_score": {self.system_score}, "scores": {self.scores}}}'

def read_txt_strip(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]

def sp_encode(model, text: List[str]):
    return [' '.join(model.encode(sentence, out_type=str)) for sentence in text]

def prepare_input_data(src: List[str], hyp: List[str], ref: List[str], metric_type: str = "seq", metric_name: str = None):
    """
    metric_type, metric_name:
        "seq": sequence-based
             "spbleu": spBLEU200
             "chrf": chrF++
             "bleurt": BLEURT20
        "qe": reference-free, quality estimation
        "reg": regression
    """
    if metric_type == "seq":
        if metric_name == "spbleu":
            sp_model = spm.SentencePieceProcessor()
            sp_model.load("functions/flores200_sacrebleu_tokenizer_spm.model")
            hyp, ref = sp_encode(sp_model, hyp), sp_encode(sp_model, ref)
        if metric_name == "chrf" or metric_name == "spbleu":
            ref = [[line] for line in ref]
            return {"predictions": hyp, "references": ref}
        if metric_name == "bleurt":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = f"./tmp_bleurt_{timestamp}"
            os.makedirs(temp_dir, exist_ok=True)
            with open(os.path.join(temp_dir, "candidates"), "w") as file1:
                for line in hyp:
                    file1.write(line + '\n')
            with open(os.path.join(temp_dir, "references"), "w") as file1:
                for line in ref:
                    file1.write(line + '\n')
            return temp_dir
    elif metric_type == "reg":
        if metric_name == "comet" or metric_name == "xcomet":
            return [{"src": src_line, "mt":hyp_line, "ref":ref_line,} for src_line, hyp_line, ref_line in zip(src, hyp, ref)]
        elif metric_name == "metricx":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = f"./tmp_metricx_{timestamp}"
            os.makedirs(temp_dir, exist_ok=True)
            absolute_dir_path = os.path.abspath(temp_dir)
            input_file_path = os.path.join(absolute_dir_path, "input.jsonl")
            with jsonlines.open(input_file_path, mode='w') as writer:
                for hypothesis, reference in zip(hyp, ref):
                    writer.write({"hypothesis": hypothesis, "reference": reference})
            return absolute_dir_path
    elif metric_type == "qe":
        if metric_name == "kiwi" or metric_name == "kiwi23":
            return [{"src": src_line, "mt":hyp_line} for src_line, hyp_line in zip(src, hyp)]
        elif metric_name == "metricx":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = f"./tmp_metricx_{timestamp}"
            os.makedirs(temp_dir, exist_ok=True)
            absolute_dir_path = os.path.abspath(temp_dir)
            input_file_path = os.path.join(absolute_dir_path, "input.jsonl")
            with jsonlines.open(input_file_path, mode='w') as writer:
                for source, hypothesis in zip(src, hyp):
                    writer.write({"source": source, "hypothesis": hypothesis})
            return absolute_dir_path

def prepare_batch_metricx_input_data(src: List[List[str]], hyp: List[List[str]], ref: List[List[str]], metric_type="reg"):
    """
    metric_type:
       "reg": need hyp and ref
       "qe": need src and hyp
    return absolute_dir_path, segments
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = f"./tmp_metric_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    absolute_dir_path = os.path.abspath(temp_dir)

    segments = [len(pair) for pair in hyp]
    
    with jsonlines.open(os.path.join(absolute_dir_path, "input.jsonl"), mode='w') as writer:
        for i in range(len(hyp)):
            if metric_type == "reg":
                for hypothesis, reference in zip(hyp[i], ref[i]):
                    writer.write({"hypothesis": hypothesis, "reference": reference})
            else:
                for source, hypothesis in zip(src[i], hyp[i]):
                    writer.write({"source": source, "hypothesis": hypothesis})
    
    return absolute_dir_path, segments

def prepare_batch_bleurt_input_data(hyp: List[List[str]], ref: List[List[str]]):
    """
    allows computing multiple pairs together to save loading cost
    return path, list[int]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = f"./tmp_bleurt_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)

    segments = [len(pair) for pair in hyp]

    with open(os.path.join(temp_dir, "candidates"), "w") as file1:
        for pair in hyp:
            for line in pair:
                file1.write(line + '\n')
    with open(os.path.join(temp_dir, "references"), "w") as file1:
        for pair in ref:
            for line in pair:
                file1.write(line + '\n')
    return temp_dir, segments



def bleurt_score(dir_path, segments):
    """
    return list[MyScore]
    """
    command = [
        "python", "-m", "bleurt.score_files",
        f"-candidate_file={os.path.join(dir_path, 'candidates')}",
        f"-reference_file={os.path.join(dir_path, 'references')}",
        "-bleurt_checkpoint=bleurt/BLEURT-20",
        f"-scores_file={os.path.join(dir_path, 'scores')}"
    ]
    subprocess.run(command, check=True)
    if os.path.exists(os.path.join(dir_path, "scores")):
        scores = list(map(float, read_txt_strip(os.path.join(dir_path, "scores"))))
        shutil.rmtree(dir_path)
    else:
        raise ValueError(f"Command fails. Please check {dir_path}")
    
    results = []
    start = 0
    for length in segments:
        if start + length > len(scores):
            raise ValueError("Segment lengths exceed the total data length.")
        results.append(MyScores(scores[start:start + length]))
        start += length
    return results


def seq_score(input_data, metric_name):
    """
    Return MyScore
    """
    assert metric_name is not None
    if metric_name == "chrf":
        scorer = CHRF(word_order=2)
        scores = [scorer.sentence_score(input_data["predictions"][i], input_data["references"][i]).score for i in range(len(input_data["predictions"]))]
    elif metric_name == "spbleu":
        scorer = BLEU(effective_order=True)
        scores = [scorer.sentence_score(input_data["predictions"][i], input_data["references"][i]).score for i in range(len(input_data["predictions"]))]

    elif metric_name == "bleurt":
        command = [
            "python", "-m", "bleurt.score_files",
            f"-candidate_file={os.path.join(input_data, 'candidates')}",
            f"-reference_file={os.path.join(input_data, 'references')}",
            "-bleurt_checkpoint=bleurt/BLEURT-20",
            f"-scores_file={os.path.join(input_data, 'scores')}"
        ]
        subprocess.run(command, check=True)
        if os.path.exists(os.path.join(input_data, "scores")):
            scores = list(map(float, read_txt_strip(os.path.join(input_data, "scores"))))
            shutil.rmtree(input_data)
        else:
            raise ValueError(f"Command fails. Please check {input_data}")
    return MyScores(scores)

def comet_score(input_data, metric_name, **kwargs):
    """
    to save time in loading model, we can load model to a gpu, then compute several pairs iterally.
    input_data: list[ pair_1, pair_2, ...]
    return: list[MyScores]
    """
    from comet import download_model, load_from_checkpoint
    model_dict = {
        "comet": "Unbabel/wmt22-comet-da",
        "xcomet": "Unbabel/XCOMET-XL",
        "kiwi": "Unbabel/wmt22-cometkiwi-da",
        "kiwi23": "Unbabel/wmt23-cometkiwi-da-xl",
    }
    model_path = download_model(model_dict[metric_name])
    model = load_from_checkpoint(model_path)
    results = []
    if "device" in kwargs.keys():
        model.to(kwargs["device"])
        for pair in input_data:
            tmp_output = model.predict(pair, batch_size=kwargs["batch_size"])
            results.append(MyScores(tmp_output.scores, tmp_output.system_score))
    elif "gpus" in kwargs.keys():
        segments = [ len(pair) for pair in input_data]
        flattened_list = [item for sublist in input_data for item in sublist]
        tmp_output = model.predict(flattened_list, batch_size=kwargs["batch_size"], gpus=kwargs["gpus"])
        scores = tmp_output.scores
        start = 0
        for length in segments:
            if start + length > len(scores):
                raise ValueError("Segment lengths exceed the total data length.")
            results.append(MyScores(scores[start:start + length]))
            start += length
    return results


def metricx_score(dir_path, metric_type, segments: List[int] = None, batch_size=1):
    model_dict = {
        "qe": "google/metricx-23-qe-large-v2p0",
        "reg": "google/metricx-23-large-v2p0",
    }
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    command = [
    "python", "-m", "metricx23.predict",
    "--tokenizer", "google/mt5-xl",
    "--model_name_or_path", f"{model_dict[metric_type]}",
    "--max_input_length", "1024",
    "--batch_size", f"{batch_size}",
    "--input_file", f"{dir_path}/input.jsonl",
    "--output_file", f"{dir_path}/output.jsonl",
    ]
    if metric_type == "qe":
        command.append("--qe")
    
    subprocess.run(command, cwd="./metricx", check=True, env=env)
    output_path = os.path.join(dir_path, "output.jsonl")
    scores = []
    if os.path.exists(output_path):
        with jsonlines.open(output_path) as reader:
            for obj in reader:
                scores.append(obj["prediction"])
        shutil.rmtree(dir_path)
    else:
        raise ValueError(f"Command fails. Please check {dir_path}")
    
    if segments is not None:
        results = []
        start = 0
        for length in segments:
            if start + length > len(scores):
                raise ValueError("Segment lengths exceed the total data length.")
            results.append(MyScores(scores[start:start + length]))
            start += length
        return results
    else:
        return MyScores(scores)

        

