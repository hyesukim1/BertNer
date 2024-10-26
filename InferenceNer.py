import pandas as pd
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import BertConfig
import torch
import json
from ConfigNer import *

id_to_label = {v: k for k, v in label_map.items()}

pretrained_model_name = "beomi/kcbert-base"
downstream_model_dir = os.path.join(os.getcwd(), "Results\\2024-09-26 14_02_42.364150-epoch=4-val_loss=0.04.ckpt")
max_seq_length=16

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name,
    do_lower_case=False,
)

fine_tuned_model_ckpt = torch.load(
    downstream_model_dir,
    map_location=torch.device("cpu")
)


pretrained_model_config = BertConfig.from_pretrained(
    pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)

model = BertForTokenClassification(pretrained_model_config)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()

filtered_label_map = {label: token for token, label in label_map.items() if label not in {0, 1, 2, 3, 4}}
def map_labels_to_tokens(tokens, predicted_tags):
    result = {}
    for token, tag in zip(tokens, predicted_tags):
        if tag != 'O' and tag not in {'[CLS]', '[SEP]', '[PAD]', '[MASK]'}:  # Exclude special tokens
            if tag not in result:
                result[tag] = token
            else:
                result[tag] += token  # Append subword tokens correctly if needed
    return result

def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=64,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[int(idx.item())] for idx in preds]
        check_la = map_labels_to_tokens(tokens, predicted_tags)
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                token_result = {
                    "token": token,
                    "predicted_tag": predicted_tag,
                    "top_prob": str(round(top_prob[0].item(), 4)),
                }

                result.append(token_result)
    return {
        "sentence": sentence,
        "result": result,
        "check": check_la
    }
# file = 'C:/Users/hyesukim/Documents/GitHub/Chatbot.Engine.me.v1/BertNer/DataNer/MakeQuestionNer.csv'
# question_datas = pd.read_csv(file)
# data = question_datas['questions'].tolist()
# inputs = data

file2 = 'C:/Users/hyesukim/Documents/GitHub/Chatbot.Engine.me.v1/Rag/DataRag/MakeQuestionRag.json'

with open(file2, 'r', encoding='utf-8') as file:
    data = json.load(file)

inputs = [entry['input'] for entry in data]

for i in inputs[:10]:
    print(i)
    a = inference_fn(i)
    print(a)

'''
라벨링 다시 하니까 성능 괜찮아짐
{'sentence': '봄에 참여할 수 있는 프로그램은 무엇인가요?', 
'result': [{'token': '봄', 'predicted_tag': 'B-TIME', 'top_prob': '0.997'}, 
{'token': '##에', 'predicted_tag': 'O', 'top_prob': '0.9994'}, 
{'token': '참여', 'predicted_tag': 'B-STATUS', 'top_prob': '0.9994'}, 
{'token': '##할', 'predicted_tag': 'O', 'top_prob': '0.9997'}, 
{'token': '수', 'predicted_tag': 'O', 'top_prob': '0.9998'}, 
{'token': '있는', 'predicted_tag': 'O', 'top_prob': '0.9998'}, 
{'token': '프로그램', 'predicted_tag': 'B-SERVICE', 'top_prob': '0.9993'}, 
{'token': '##은', 'predicted_tag': 'O', 'top_prob': '0.9997'}, 
{'token': '무엇인가', 'predicted_tag': 'O', 'top_prob': '0.9998'}, 
{'token': '##요', 'predicted_tag': 'O', 'top_prob': '0.9998'}, 
{'token': '?', 'predicted_tag': 'O', 'top_prob': '0.9997'}]}
'''
k=10