import pandas as pd
import random
import os
import numpy as np
from pathlib import Path

def sample_files(language, num_samples=102, num_iterations=10):
    current_dir = Path.cwd()
    input_files = [
      
        current_dir/"scale"/"merged_result"/f"en-{language}-merge-1.tsv",
        current_dir/"scale"/"merged_result"/f"en-{language}-merge-2.tsv",
        current_dir/"scale"/"merged_result"/f"en-{language}-merge-3.tsv",
        current_dir/"scale"/"merged_result"/f"en-{language}-merge-4.tsv",
        current_dir/"scale"/"merged_result"/f"en-{language}-merge-5.tsv",

    ]
    random.seed(42)
    np.random.seed(42)
    for input_file in input_files:

        file_number = Path(input_file).stem.split("-")[-1]
        df = pd.read_csv(input_file, sep="\t")


        for iteration in range(1, num_iterations + 1):
            sampled_df = df.sample(n=num_samples, random_state=random.randint(0, 10000))
            sampled_df = sampled_df.replace({r"<v>": "", r"</v>": ""}, regex=True)
            sampled_df = sampled_df.replace({r"\s{2,}": " "}, regex = True)

            output_file = current_dir/"scale"/"merged_result"/f"en-{language}-merge-{file_number}_{iteration}.tsv"
            sampled_df.to_csv(output_file, sep="\t", index=False, quoting=3)
            print(f"Saved: {output_file}")



sample_files("zh")
sample_files("lo")
sample_files("ja")
sample_files("vi")
sample_files("id")
sample_files("de")
sample_files("es")
sample_files("si")
sample_files("fr")
