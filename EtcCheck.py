# 라벨 수 체크: 임벨런싱 문제 확인용
# 데이터 어떻게 1천개로 구축할 수 있을지 고민

import os
import pandas as pd
from transformers import BertConfig, BertForTokenClassification,  BertTokenizer
from ConfigNer import *

arg = Arguments

max_length = arg.max_seq_length
NER_CLS_TOKEN = "[CLS]"
NER_SEP_TOKEN = "[SEP]"
NER_PAD_TOKEN = "[PAD]"
NER_MASK_TOKEN = "[MASK]"


def token_rule(sentence_list):
    sentence = [NER_CLS_TOKEN] + sentence_list + [NER_SEP_TOKEN]
    pad_length = max(max_length - len(sentence), 0)
    pad_sequence = [NER_PAD_TOKEN] * pad_length
    sentence += pad_sequence
    return sentence

def get_label_for_token(token, prev_label, prev_token, later_token):
    # tokenized_variables의 각 카테고리 및 토큰과 비교
    for category, token_lists in tokenized_variables["ko_bert_tokens_variables"][0].items():
        for token_list in token_lists:
            if '##' not in token and token in token_list:  # 해당 토큰을 저장하고 싶은데
                if prev_token == "##할" and token == "수":
                    return "O"
                elif token == "무":
                    if later_token in token_list:
                        return f"B-{category}"
                elif token == "가":
                    if later_token in token_list:
                        return f"B-{category}"
                elif token == "어":
                    if later_token in token_list:
                        return f"B-{category}"
                elif token == "시":
                    if later_token in token_list:
                        return f"B-{category}"
                elif token == "단":
                    return f"I-TARGET"

                else:
                    return f"B-{category}"

            elif '##' in token and token in token_list:  # 첫 번째 이후 토큰일 경우, 이전 토큰과 같은 카테고리일 때만 I- 태그
                if prev_token in token_list and prev_label == f"B-{category}":
                    return f"I-{category}"
                elif prev_token in token_list and prev_label == f"I-{category}":
                    return f"I-{category}"
                else:
                    "O"
    return "O"

def label_tokens(tokens):
    labels = []
    prev_label = "O"
    prev_token = "O"
    for ind, token in enumerate(tokens):
        if ind < len(tokens) - 1:
            next_token = tokens[ind + 1]
        else:
            next_token = None

        label = get_label_for_token(token, prev_label, prev_token, next_token)
        labels.append(label)
        prev_label = label
        prev_token = token

    original = token_rule(tokens)
    label1 = token_rule(labels)
    return original, label1

data = pd.read_csv(os.getcwd()+'/DataNer/MakeQuestionNer.csv')
all_questions = data['questions'].tolist()

input_idss = []
attention_masks = []
token_type_idsss = []
label_idss = []
tokenss = []

for sentence in all_questions:
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")

    tokens = tokenizer.tokenize(sentence)
    tokenss.append(tokens)
    ori, la = label_tokens(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(ori)
    attention_mask = [1 if token != 0 else 0 for token in input_ids]
    token_type_ids = [0] * max_length
    label_ids = [label_map[label] for label in la]

    input_idss.append(input_ids)
    attention_masks.append(attention_mask)
    token_type_idsss.append(token_type_ids)
    label_idss.append(label_ids)

def label_count(label_id, label_map):
    flat_data = [item for sublist in label_id for item in sublist]
    count_series = pd.Series(flat_data).value_counts()
    inverse_label_map = {v: k for k, v in label_map.items()}
    mapped_counts = count_series.rename(index=inverse_label_map)
    return mapped_counts

label_num = label_count(label_idss, label_map)
print(label_num)

k=10