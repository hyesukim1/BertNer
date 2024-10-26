import random
import pandas as pd
from transformers import BertConfig, BertForTokenClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# 기타 컨피그 및 아규먼트 세팅
from ConfigNer import *
# 모델 훈련 및 평가 클래스, 학습 전반 모듈
from ModelNer import *
# 라벨 데이터 생성 및 데이터 메모리 형태로 들고 있음, Dataset 역할
from TokenLabelNer import *
# 데이터 콜레이터는 딕셔너리 객체를 배치로 만들어줌
from DataUtils import *

arg = Arguments

# 사전 학습 완료된 BERT 모델에 개체명 인식을 위한 테스크 모듈이 덧붙여짐
# kcbert-base 417M, kcbert-large 1.2G
pretrained_model_config = BertConfig.from_pretrained(
    arg.pretrained_model_name,
    num_labels=23, # 내 라벨 갯수
)

model = BertForTokenClassification.from_pretrained(
        arg.pretrained_model_name,
        config=pretrained_model_config,
)

data = pd.read_csv(os.getcwd()+'/DataNer/MakeQuestionNer.csv')
all_questions = data['questions'].tolist()

Train = random.sample(all_questions, int(0.8 * len(all_questions)))
train_data = NerLabelProcessor(Train, arg.pretrained_model_name, tokenized_variables["ko_bert_tokens_variables"][0], label_map, arg.max_seq_length)

Valid = list(set(all_questions)-set(Train))
valid_data = NerLabelProcessor(Valid, arg.pretrained_model_name, tokenized_variables["ko_bert_tokens_variables"][0], label_map, arg.max_seq_length)

train_dataloader = DataLoader(
    train_data,
    batch_size=arg.batch_size,
    sampler=RandomSampler(train_data, replacement=False),
    collate_fn=data_collator,
    drop_last=False
)

val_dataloader = DataLoader(
    valid_data,
    batch_size=arg.batch_size,
    sampler=SequentialSampler(valid_data),
    collate_fn=data_collator,
    drop_last=False
)

task = NERTask(model, arg)

trainer = get_trainer(arg)
trainer.fit(task, train_dataloader, val_dataloader)

k=10