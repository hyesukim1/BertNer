import torch
from dataclasses import dataclass
from typing import List, Optional
from transformers import BertTokenizer

from ConfigNer import *


@dataclass
class NERFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None

class NerLabelProcessor:
    def __init__(self, sentences, tokenizer_path, tokenized_variables, label_map, max_length):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.tokenized_variables = tokenized_variables
        self.max_length = max_length
        self.NER_CLS_TOKEN = "[CLS]"
        self.NER_SEP_TOKEN = "[SEP]"
        self.NER_PAD_TOKEN = "[PAD]"
        self.NER_MASK_TOKEN = "[MASK]"
        self.label_map = label_map
        self.sentences = sentences


    def token_check(self):
        return self.tokenizer

    def tokenize_sentence(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_label_for_token(self, token, prev_label, prev_token, later_token):
        # tokenized_variables의 각 카테고리 및 토큰과 비교
        for category, token_lists in self.tokenized_variables.items():
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

    def token_rule(self, sentence_list):
        sentence = [self.NER_CLS_TOKEN] + sentence_list + [self.NER_SEP_TOKEN]
        pad_length = max(self.max_length - len(sentence), 0)
        pad_sequence = [self.NER_PAD_TOKEN] * pad_length
        sentence += pad_sequence
        return sentence

    def label_tokens(self, tokens):
        labels = []
        prev_label = "O"
        prev_token = "O"
        for ind, token in enumerate(tokens):
            if ind < len(tokens) - 1:
                next_token = tokens[ind + 1]
            else:
                next_token = None

            label = self.get_label_for_token(token, prev_label, prev_token, next_token)
            labels.append(label)
            prev_label = label
            prev_token = token

        original = self.token_rule(tokens)
        label1 = self.token_rule(labels)
        return original, label1

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        tokens = self.tokenize_sentence(sentence)
        ori, la = self.label_tokens(tokens)
        input_ids = self.convert_tokens_to_ids(ori)
        attention_mask = [1 if token != 0 else 0 for token in input_ids]
        token_type_ids = [0] * self.max_length
        label_ids = [self.label_map[label] for label in la]

        # features = NERFeatures(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     label_ids=label_ids
        # )
        # return features

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'label_ids': torch.tensor(label_ids)
        }


    def get_labels(self):
        return self.label_map

    def get_sentence(self, idx):
        sentence = self.sentences[idx]
        return sentence

# 참고: https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891
