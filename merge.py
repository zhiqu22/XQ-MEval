import pandas as pd
import random
import itertools
from pathlib import Path

zh_Addition_ids = [5, 25, 26, 105, 112, 119, 149, 163, 191, 196]
zh_Omission_ids = [89, 118, 126, 130]
zh_Mistranslation_ids = [85, 107, 163]
zh_Untranslated_ids = [6, 18, 19, 24, 29, 41, 52, 71, 72, 75, 80, 83, 86, 100, 110, 130, 131, 136, 174, 183, 185, 186, 203]
lo_Addition_ids = [16, 41, 48, 69, 122, 139, 179, 200]
lo_Omission_ids = [24, 36, 53, 67, 68, 69, 104, 108, 118, 121, 125, 185, 200]
lo_Mistranslation_ids = [23, 41, 143, 156]
lo_Untranslated_ids = [3, 13, 30, 46, 49, 63, 65, 67, 79, 80, 81, 94, 95, 100, 112, 123, 126, 131, 132, 134, 141, 149, 151, 159, 178, 188, 47, 50, 91, 106, 127, 137]
ja_Addition_ids = [128, 193, 162, 163, 166, 11, 43, 107, 112, 124]
ja_Omission_ids = [130, 165, 166, 167, 169, 44, 82, 182]
ja_Mistranslation_ids = [5, 6]
ja_Untranslated_ids = [64, 131, 69, 10, 202, 204, 146, 22, 23, 24, 95, 100, 165, 103, 104, 105, 49, 185, 186, 59, 60]
vi_Addition_ids = [3, 65, 110]
vi_Omission_ids = [3, 27, 30, 63, 105]
vi_Mistranslation_ids = [66, 80, 157, 179]
vi_Untranslated_ids = [5, 14, 24, 30, 31, 35, 38, 47, 49, 52, 56, 72, 80, 82, 86, 98, 99, 104, 112, 117, 122, 128, 131, 132, 133, 136, 146, 166, 180, 182, 191, 203, 204]
id_Addition_ids = [41, 61, 71, 73, 75, 115, 161, 172]
id_Omission_ids = [40, 71, 86, 141, 165, 169, 193]
id_Mistranslation_ids = [7, 137, 169]
id_Untranslated_ids = [29, 30, 46, 47, 51, 63, 65, 68, 71, 72, 124, 131, 139, 149, 165, 188]
fr_Addition_ids = [48, 3, 121, 62]
fr_Omission_ids = [130, 100, 15, 143, 182, 151, 88, 123]
fr_Mistranslation_ids = [101, 69, 201, 47, 23, 151, 125, 158]
fr_Untranslated_ids = [128, 65, 193, 68, 196, 136, 203, 143, 150, 23, 94, 97, 35, 168, 104, 47, 48, 116, 185, 59, 60]
es_Addition_ids = [131, 132, 199, 202, 45, 18, 51, 95]
es_Omission_ids = [101, 197, 7, 201, 73, 174, 158]
es_Mistranslation_ids = [4, 137, 77, 81, 17, 19, 179]
es_Untranslated_ids = [4, 132, 196, 72, 202, 203, 76, 13, 204, 17, 22, 94, 160, 162, 100, 167, 187, 47, 176, 180, 123, 60, 62]
si_Addition_ids = [47, 49, 56, 142]
si_Omission_ids = [30, 36, 45, 67, 68, 77, 94, 117, 120, 136, 140, 142, 155, 162, 175, 187]
si_Mistranslation_ids = [4, 36, 80, 82, 162, 188, 192]
si_Untranslated_ids = [3, 10, 24, 35, 50, 65, 67, 68, 72, 82, 93, 94, 95, 103, 118, 131, 134, 154, 164, 183, 195, 197, 203]
de_Addition_ids = [18]
de_Omission_ids = [100, 4, 102, 201, 123, 77, 142, 23, 120, 187]
de_Mistranslation_ids = [96, 101, 169, 173, 17, 147, 85, 86, 124, 157]
de_Untranslated_ids = [129, 131, 10, 141, 18, 24, 29, 158, 38, 43, 45, 54, 58, 62, 63, 64, 69, 71, 72, 92, 104]


def concat_files(language):
    current_dir = Path(__file__).parent
    file_configs = [
        {
            "file_path": current_dir / "results" / f"en-{language}-Addition.tsv",
            "excluded_count_ids": f"{language}_Addition_ids",  
        },
        {
            "file_path": current_dir / "results" / f"en-{language}-Untranslated.tsv",
            "excluded_count_ids": f"{language}_Untranslated_ids",  
        },
        {
            "file_path": current_dir / "results" / f"en-{language}-Mistranslation.tsv",
            "excluded_count_ids": f"{language}_Mistranslation_ids",  
        },
        {
            "file_path": current_dir / "results" / f"en-{language}-Omission.tsv",
            "excluded_count_ids": f"{language}_Omission_ids", 
        },
    ]

    dataframes = []
    for config in file_configs:
        file_path = config["file_path"]
        excluded_count_ids = globals().get(config["excluded_count_ids"])
        df = pd.read_csv(file_path, sep="\t", quoting=3)
        df = df[~df['count_id'].isin(excluded_count_ids)]
        dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True)
    output_file = f"en-{language}.tsv"
    merged_df.to_csv(output_file, sep="\t", index=False, quoting=3)


def find_error_index(ref, mt):
   
    if "<v>" in mt and "</v>" in mt:

        v_start = mt.index("<v>")
        v_end = mt.index("</v>") + len("</v>")
        
        before_v_context = mt[:v_start]  
        after_v_context = mt[v_end:] 
         
        start_index = None
        end_index = None
        
        try:
            if before_v_context: 
                start_index = ref.find(before_v_context) + len(before_v_context) 
            else:
                start_index = 0 
            if after_v_context:  
                end_index = ref.rfind(after_v_context) + 1 
            else:
                end_index = len(ref) + 1  
            if start_index is not None and end_index is not None:
                return f"{start_index }-{end_index}" 
            else:
                return None
        except ValueError:
            return None
    else:
        return None
        
def find_error_index_words(ref, mt):
   
    if "<v>" in mt and "</v>" in mt:

        v_start = mt.index("<v>")
        v_end = mt.index("</v>") + len("</v>")
        
        before_v_context = mt[:v_start].strip().split()
        
        after_v_context = mt[v_end:].strip().split()
        
        ref_words = ref.split()
 
         
        start_index = None
        end_index = None
        
        
        if before_v_context: 
           ref_index = 0  
           for word in before_v_context:  
               if ref_index < len(ref_words) and ref_words[ref_index] == word:
                  start_index = ref_index  
                  ref_index += 1 
           start_index = ref_index
        else:
            start_index = 0   
        

        if after_v_context:
           ref_index = len(ref_words) - 1 
           for word in reversed(after_v_context):  
               if ref_index >= 0 and ref_words[ref_index] == word:
                  end_index = ref_index 
                  ref_index -= 1  
           end_index = ref_index + 2
        else:
            end_index = len(ref_words) + 1
     
        if start_index is not None and end_index is not None:
           return f"{start_index}-{end_index}" 
                

def get_error_spans_file(language):
    input_file = f"en-{language}.tsv"
    output_file = f"en-{language}-error-span.tsv"
    
    df = pd.read_csv(input_file, sep="\t", quoting=3)
    if language in ["zh", "ja", "lo"]:
        df['error_span'] = df.apply(lambda row: find_error_index(row['ref'], row['mt']), axis=1)
    else:
        df['error_span'] = df.apply(lambda row: find_error_index_words(row['ref'], row['mt']), axis=1)
    df = df[df['error_span'].notnull()]
    df.to_csv(output_file, sep="\t", index=False, quoting=3)


def non_overlapping(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    return (end1 <= start2 or end2 <= start1)

def find_all_non_overlapping_combinations(spans, n):
    all_combinations = itertools.combinations(spans, n)
    non_overlapping_combinations = []
  
    for combination in all_combinations:
        overlapping = False
        for i in range(len(combination)):
            for j in range(i + 1, len(combination)):
                if not non_overlapping(combination[i], combination[j]):  
                    overlapping = True
                    break
            if overlapping:
                break
        if not overlapping:
            non_overlapping_combinations.append(combination)
    
    return non_overlapping_combinations




def merge_n_spans(ref, mts, spans):
    merged_content = ""
    current_pos = 0  
  
    for i, span in enumerate(spans):
        mt = mts[i]
        v_start = mt.index("<v>") + len("<v>")  
        v_end = mt.index("</v>")  
        v_content = mt[v_start:v_end]  

        before_span = ref[current_pos:span[0]]
        merged_content += f"{before_span}<v>{v_content}</v>"

        current_pos = span[1] - 1

    merged_content += ref[current_pos:]
    return merged_content

def merge_n_spans_words(ref, mts, spans):
    merged_content = ""
    current_pos = 0
    ref_words = ref.split()
    for i, span in enumerate(spans):
        mt = mts[i]
        v_start = mt.index("<v>") + len("<v>")  
        v_end = mt.index("</v>")  
        v_content_words = mt[v_start:v_end].strip().split()
        v_content = " ".join(v_content_words)
        
        
        before_span = " ".join(ref_words[current_pos:span[0]])
    
        merged_content += f"{before_span} <v>{v_content}</v> "
        

        current_pos = span[1] - 1
    merged_content += " ".join(ref_words[current_pos:])

    return merged_content

def merge_error_spans(language,n):

    input_file = f"en-{language}-error-span.tsv" 
    output_file = f"en-{language}-merge-{n}.tsv"
    
    df = pd.read_csv(input_file, sep="\t", quoting=3)

    results = []
    
    for segment_id, group in df.groupby('segment_id'):
        spans = []
        mts = []
        ref = group.iloc[0]['ref']  
        src = group.iloc[0]['src']
        
        for _, row in group.iterrows():
            start, end = map(int, row['error_span'].split('-'))
            spans.append((start, end))
            mts.append(row['mt'])
        
        all_combinations = find_all_non_overlapping_combinations(spans,n)
        
        for span_group in all_combinations:
            sorted_span_group = tuple(sorted(span_group, key=lambda x: x[0]))

            mt_group = [mts[spans.index(span)] for span in sorted_span_group]
            if language in ["zh","ja", "lo"]:
               merged_mt = merge_n_spans(ref, mt_group, sorted_span_group)
            else:
               merged_mt = merge_n_spans_words(ref, mt_group, sorted_span_group)
                   
            
            results.append({
                'segment_id': segment_id,
                'spans': "; ".join([f"{span[0]}-{span[1]}" for span in sorted_span_group]),
                'src': src,
                'ref': ref,
                'merged_mt': merged_mt
            })
        
        
    result_df = pd.DataFrame(results)
    result_df = result_df.replace({"\n": " ", "\t": " "}, regex=True).map(lambda x: x.strip() if isinstance(x, str) else x)

    result_df.to_csv(output_file, sep="\t", index=False, quoting = 3)




languages = ["zh", "lo", "ja", "vi", "id", "fr", "es", "si", "de"]
tgt_lang = languages[1]
concat_files(tgt_lang)
get_error_spans_file(tgt_lang)
merge_error_spans(tgt_lang,2)
