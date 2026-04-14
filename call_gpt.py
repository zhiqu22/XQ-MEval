import functions.prompt_tools as prompt_tools
import functions.score_tools as score_tools
from openai import OpenAI
import os
import logging
import pandas as pd
import re

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
logging.basicConfig(filename=os.path.join("logs", "call_gpt.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

languages = ["zh", "de", "ja"]

indices = [
    2, 10, 14, 27, 30, 31, 51, 54, 63, 86, 87,
    101, 114, 116, 119, 127, 131, 141, 152, 155,
    162, 165, 166, 196, 204, 205, 215, 237, 247,
    255, 265, 274, 288, 325, 326, 331, 336, 355,
    365, 371, 372, 373, 376, 377, 412, 415, 417,
    441, 443, 462, 470, 474, 476, 481, 482, 484,
    487, 495, 497, 498, 503, 504, 507, 530, 532,
    544, 550, 565, 566, 567, 591, 592, 598, 602,
    604, 615, 635, 656, 660, 665, 668, 673, 678,
    682, 734, 761, 766, 793, 801, 822, 849, 857,
    925, 926, 928, 932, 936, 944, 952, 974, 983, 1001
]

data = {
    "language": [],
    "count_id": [],
    "segment_id": [],
    "error_type": [],
    "error_position": [],
    "src": [],
    "ref": [],
    "mt": [],
    "reject": []
}

client = OpenAI(
    organization = os.getenv("OPENAI_ORGANIZATION_KEY"),
    api_key = os.getenv("OPENAI_API_KEY"),                                                                                 
)

def call(instruction, temperature=0.5):                                                                                                                          
    response = client.chat.completions.create(
        # grammar is 11-20
        # model='gpt-4o-2024-11-20',
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "user", "content": instruction},
        ],
        temperature = temperature,
        seed = 42,
        max_tokens = 2000,
    )
    return response

def execute(id, language, error_type, position, src, tgt, ref) -> str:
    instruction = prompt_tools.generate_prompt(error_type, position, tgt, src=src, tgt_language=language)

    if error_type != "Grammar":
        response = call(instruction)
    else:
        response = call(instruction, temperature=1.0)

    finish_reason = response.choices[0].finish_reason
    if finish_reason != "stop":
        logging.error(f"Fail caused by API: {language}-{id}-{error_type}-{position}")
        logging.error(f"Fail info: {response.choices[0]}")
        return None
    
    raw_result = remove_double_asterisks(response.choices[0].message.content)
    result = None

    # confirm the possible prefix
    prefixes = ["Updated target sentence:", "Updated sentence:", "Updated Sentence:", "updated sentence:", "updated sentence is:", 
                "Output:", "Updated sentence with the disrupted sub-part:", "updated sentence", "Final output:", "final output:", "Final Output:", 
               "Finally, output the updated sentence:"]
    
    for prefix in prefixes:
        if prefix in raw_result:
            start_index = raw_result.rfind(prefix) + len(prefix)
            raw_result = raw_result[start_index:].strip()
            result = raw_result
    
    # whether we successfully turncate the output
    if result is None:
        logging.error(f"Fail by invalid result: {language}-{id}-{error_type}-{position}")
        logging.error(f"Fail info: {response.choices[0]}")
        return None, response.choices[0]
    else:
        if "\n" in result:
            sub_results = result.split("\n")
            result = max(sub_results, key=len).strip()
    
    result = remove_position_markers(result)

    # whether the result is a complete sentence
    if len(result) < 0.6 * len(ref) or len(result) > 1.4 * len(ref):
        # zh and ja has shorter characters in general.
        if not (error_type == "Untranslated" and language in ["zh", "ja"]):
            logging.error(f"Fail by incomplete result: {language}-{id}-{error_type}-{position}")
            logging.error(f"Fail info: {response.choices[0]}")
            return None, response.choices[0]

    logging.info(f"Success: {language}-{id}-{error_type}-{position}")

    return result, response.choices[0]

def character_level_label_error(sentence, ref):
    start_diff = 0
    while start_diff < len(sentence) and start_diff < len(ref) and sentence[start_diff] == ref[start_diff]:
        start_diff += 1
    
    end_diff_sentence = len(sentence)
    end_diff_ref = len(ref)
    while (
        end_diff_sentence > start_diff and
        end_diff_ref > start_diff and
        sentence[end_diff_sentence - 1] == ref[end_diff_ref - 1]
    ):
        end_diff_sentence -= 1
        end_diff_ref -= 1
    
    if start_diff == 0 and end_diff_sentence == len(sentence):
        # if <v> and </v> occurs at the head and the end, respectively and simultaneously,
        # the sentence is wrapped by some notations, e.g., " and '.
        # We want to remove the warp, then recursively call this function. 
        sentence = sentence[1:-1]
        # recursion
        return character_level_label_error(sentence, ref)
    
    return sentence[:start_diff] + "<v>" + sentence[start_diff:end_diff_sentence] + "</v>" + sentence[end_diff_sentence:]

def word_level_label_error(sentence, ref):
    if sentence == ref:
        return None
    sentence_words = sentence.split()
    ref_words = ref.split()

    start_diff = 0
    while (start_diff < len(sentence_words) and
           start_diff < len(ref_words) and
           sentence_words[start_diff] == ref_words[start_diff]):
        start_diff += 1

    end_diff_sentence = len(sentence_words)
    end_diff_ref = len(ref_words)
    while (end_diff_sentence > start_diff and
           end_diff_ref > start_diff and
           sentence_words[end_diff_sentence - 1] == ref_words[end_diff_ref - 1]):
        end_diff_sentence -= 1
        end_diff_ref -= 1
    
    if start_diff == 0 and end_diff_sentence == len(sentence_words):
        # if <v> and </v> occurs at the head and the end, respectively and simultaneously,
        # the sentence is wrapped by some notations, e.g., " and '.
        # We want to remove the warp, then recursively call this function. 
        sentence = sentence[1:-1]
        # recursion
        return word_level_label_error(sentence, ref)
    # combine sentence
    result = (
        " ".join(sentence_words[:start_diff]) + 
        " <v>" + 
        " ".join(sentence_words[start_diff:end_diff_sentence]) + 
        "</v>" + 
        " " + " ".join(sentence_words[end_diff_sentence:])
    )
    return result.strip()

def label_error(sentence, ref, tgt_lang):
    if tgt_lang in ["zh", "ja"]:
        return character_level_label_error(sentence, ref)
    else:
        return word_level_label_error(sentence, ref)
    
def remove_position_markers(result):
    pattern = fr'<({"|".join(prompt_tools.position_list)})>|</({"|".join(prompt_tools.position_list)})>'
    return re.sub(pattern, '', result)

def remove_double_asterisks(input_string):
    return re.sub(r'\*\*', '', input_string)

def post_edit(result: str, tgt: str, tgt_lang: str) -> str:
    # 1. remove the position maker, if the position maker exists.
    result = remove_position_markers(result)

    # 2. add the error maker, if the error maker does not exist.
    if re.search(r'<v>.*?</v>', result):
        return result
    else:
        return label_error(result, tgt, tgt_lang)



tgt_lang = languages[1]
error_type = prompt_tools.error_type_list[1]
source_path = f"floresp-v2.0-rc.3/devtest/devtest.{prompt_tools.translate_language_code('en', 'iso', 'flores')}"
target_path = f"floresp-v2.0-rc.3/devtest/devtest.{prompt_tools.translate_language_code(tgt_lang, 'iso', 'flores')}"

count = 0
for idx in indices:
    src = score_tools.read_txt_strip(source_path)[idx]
    tgt = score_tools.read_txt_strip(target_path)[idx]
    if tgt_lang in ['zh', 'ja']:  
        elements = list(tgt)
        total = len(elements)
        part_length = total // 2
        head_part = "".join(elements[:part_length])
        end_part = "".join(elements[part_length:])
        tgt_with_position_makers = f"<head>{head_part}</head><end>{end_part}</end>"
    else:
        elements = tgt.split()      
        total = len(elements)
        part_length = total // 2
        head_part = " ".join(elements[:part_length])
        end_part = " ".join(elements[part_length:])
        tgt_with_position_makers = f"<head>{head_part}</head> <end>{end_part}</end>"
    
    for position in prompt_tools.position_list:
        count += 1
        data["language"].append(tgt_lang)
        data["count_id"].append(count)
        data["segment_id"].append(idx)
        data["error_type"].append(error_type)
        data["error_position"].append(position)
        data["src"].append(src)
        data["ref"].append(tgt)
        data["reject"].append(" ")
        
        result, gpt_output = execute(idx, tgt_lang, error_type, position, src, tgt=tgt_with_position_makers, ref=tgt)
        
        if result is None:
            data["mt"].append("please check log to see the error information") 
            continue

        # post-edit
        result = post_edit(result, tgt, tgt_lang)
        if result is None:
            logging.error(f"Fail by no error in post edit: {tgt_lang}-{idx}-{error_type}-{position}")
            logging.error(f"Fail info: {gpt_output}")
            result = "please check log to see the error information"
        data["mt"].append(result)
        
df = pd.DataFrame(data)
df.to_csv(f"results/en-{tgt_lang}-{error_type}.tsv", sep="\t", index=False, quoting=3)

