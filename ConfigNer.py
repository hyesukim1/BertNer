import os
class Arguments:
    pretrained_model_name = "beomi/kcbert-base"
    downstream_task_name = "named-entity-recognition"
    downstream_corpus_name = "ner"
    downstream_model_dir = os.getcwd() + "/Results"
    batch_size = 8 # 라벨 갯수 보고 적절히 조절
    learning_rate = 5e-5
    max_seq_length = 16 # 내 문장이 길지 않음
    epochs = 5
    seed = 7
    NER_PAD_ID = 2
    save_top_k = 1
    monitor = "min val_loss"
    test_mode = False
    fp16 = False

label_map = {
            '[CLS]': 0,
            '[SEP]': 1,
            '[PAD]': 2,
            '[MASK]': 3,
            'O': 4,
            'B-SERVICE': 5,
            'I-SERVICE': 6,
            'B-TIME': 7,
            'I-TIME': 8,
            'B-TARGET': 9,
            'I-TARGET': 10,
            'B-PRICE': 11,
            'I-PRICE': 12,
            'B-PLACE': 13,
            'I-PLACE': 14,
            'B-CONTENT': 15,
            'I-CONTENT': 16,
            'B-STATUS': 17,
            'I-STATUS': 18,
            'B-INFO': 19,
            'I-INFO': 20,
            'B-HOW': 21,
            'I-HOW': 22
        }

tokenized_variables ={
    "multi_lingual_bert_tokens_variables" : [{
        'SERVICE': [['프로', '##그램']],
       'TIME': [['봄'], ['여', '##름'], ['가', '##을'], ['겨', '##울'], ['여', '##름', '방', '##학'],
                ['겨', '##울', '방', '##학'], ['이', '##번', '달'], ['이', '##달'], ['이', '##번', '주'],
                ['이', '##주'], ['1월'], ['2월'], ['3월'], ['4월'], ['5월'], ['6월'], ['7월'], ['8월'],
                ['9월'], ['10월'], ['11월'], ['12월'], ['오', '##전'], ['오', '##후'], ['휴', '##일'],
                ['주', '##말'], ['평', '##일'], ['월', '##요일'], ['화', '##요일'], ['수', '##요일'],
                ['목', '##요일'], ['금', '##요일'], ['토', '##요일'], ['일', '##요일'], ['오', '##늘'],
                ['요', '##즘'], ['주', '##말', '##마다'], ['언', '##제'], ['몇', '##시', '##에'],
                ['기', '##간'], ['시', '##간'], ['방', '##학']],
       'TARGET': [['다', '##문', '##화'], ['가', '##정'], ['부', '##모'], ['아', '##이'],
                  ['청', '##소', '##년'], ['성', '##인'], ['가', '##족'], ['어', '##른'],
                  ['초', '##등', '##학', '##생'], ['중', '##학', '##생'], ['고', '##등', '##학', '##생'],
                  ['대학', '##생'], ['장', '##애', '##인'], ['어', '##르', '##신'],
                  ['다', '##문', '##화', '가', '##정'], ['청', '##년'], ['유', '##아'], ['어린', '##이'],
                  ['단', '##체'], ['부', '##모', '##와', '아', '##이'], ['가', '##족', '단', '##위']],
       'PRICE': [['무', '##료'], ['유', '##료']],
       'PLACE': [['대', '##면'], ['온', '##라', '##인'], ['야', '##외'], ['실', '##내'], ['실', '##회'],
                 ['박', '##물', '##관'], ['공', '##원'], ['종', '##로', '##구'], ['중', '##구'],
                 ['용', '##산', '##구'], ['성', '##동', '##구'], ['광', '##진', '##구'],
                 ['동', '##대', '##문', '##구'], ['중', '##랑', '##구'], ['성', '##북', '##구'],
                 ['강', '##북', '##구'], ['도', '##봉', '##구'], ['노', '##원', '##구'], ['은', '##평', '##구'],
                 ['서', '##대', '##문', '##구'], ['마', '##포', '##구'], ['양', '##천', '##구'],
                 ['강', '##서', '##구'], ['구', '##로', '##구'], ['금', '##천', '##구'],
                 ['영', '##등', '##포', '##구'], ['동', '##작', '##구'], ['관', '##악', '##구'],
                 ['서', '##초', '##구'], ['강', '##남', '##구'], ['송', '##파', '##구'], ['강', '##동', '##구']],
       'CONTENT': [['교', '##육'], ['전', '##시'], ['산', '##림'], ['여', '##가'], ['공', '##원'], ['문', '##화'],
                   ['행', '##사'], ['역', '##사'], ['전', '##시', '##회']],
       'STATUS': [['예', '##약'], ['진', '##행'], ['종', '##료'], ['끝'], ['시', '##작'], ['등', '##록'],
                  ['신', '##청'], ['진', '##행', '##중'], ['추', '##천'], ['취', '##소'], ['참', '##여']],
       'INFO': [['링', '##크'], ['전', '##화']],
       'HOW': [['어', '##디', '##서'], ['어', '##디', '##에'], ['어', '##떻', '##게'], ['어떤', '##게'],
               ['어', '##디'], ['무', '##엇']]}],


    "ko_bert_tokens_variables": [{
        'SERVICE': [['프로그램']],
        'TIME': [['봄'], ['여름'], ['가을'], ['겨울'], ['여름', '방학'], ['겨울', '방학'], ['이번', '달'], ['이', '##달'],
                 ['이번', '주'], ['이주'], ['1월'], ['2월'], ['3월'], ['4월'], ['5월'], ['6월'], ['7월'], ['8월'], ['9월'],
                 ['10월'], ['11', '##월'], ['12월'], ['오전'], ['오후'], ['휴', '##일'], ['주말'], ['평', '##일'],
                 ['월', '##요일'], ['화', '##요일'], ['수요', '##일'], ['목', '##요일'], ['금', '##요일'], ['토요일'],
                 ['일', '##요일'], ['오늘'], ['요즘'], ['주말', '##마다'], ['언제'], ['몇', '##시에'], ['기간'], ['시간'],
                 ['시간이'], ['이', '##달에'], ['4월에'], ['11', '##월에'], ['휴', '##일에'], ['평', '##일에'],
                 ['수요', '##일에'], ['주말에']],
        'TARGET': [['청소년'], ['성인'], ['가족'], ['어른'], ['초등학생'], ['중학생'], ['고등학생'], ['대학생'], ['장애인'],
                   ['어르신'], ['다문화', '가정'], ['청년'], ['유아'], ['어린이'], ['단체'], ['부모', '##와', '아이'],
                   ['가족', '단', '##위'], ['성인이'], ['가족이'], ['어른이'], ['다문화', '가정이'], ['다문화', '가정을'],
                   ['단체가'], ['아이가'], ['아이를'], ['아이'], ['가족', '단', '##위'], ['가족', '단', '##위로'],
                   ['가족', '단', '##위를'], ['가족을']],
        'PRICE': [['무료'], ['무료로'], ['유', '##료'], ['유', '##료로']],
        'PLACE': [['야외'], ['실', '##내'], ['실', '##외'], ['박', '##물', '##관'], ['공', '##원'], ['종로', '##구'],
                  ['중', '##구'], ['용', '##산', '##구'], ['성', '##동', '##구'], ['광진', '##구'], ['동', '##대문', '##구'],
                  ['중', '##랑', '##구'], ['성', '##북', '##구'], ['강', '##북', '##구'], ['도', '##봉', '##구'],
                  ['노', '##원', '##구'], ['은', '##평', '##구'], ['서', '##대문', '##구'], ['마포', '##구'],
                  ['양', '##천', '##구'], ['강', '##서', '##구'], ['구', '##로', '##구'], ['금', '##천', '##구'],
                  ['영', '##등', '##포', '##구'], ['동작', '##구'], ['관', '##악', '##구'], ['서초', '##구'],
                  ['강남구'], ['송', '##파', '##구'], ['강', '##동', '##구'], ['실', '##내에서'], ['실', '##외에', '##서'],
                  ['박', '##물', '##관에', '##서'], ['공', '##원에서'], ['종로', '##구에서'], ['중', '##구에서'],
                  ['용', '##산', '##구에서'], ['성', '##동', '##구에서'], ['광진', '##구에서'], ['동', '##대문', '##구에서'],
                  ['중', '##랑', '##구에서'], ['성', '##북', '##구에서'], ['강', '##북', '##구에서'], ['도', '##봉', '##구에서'],
                  ['노', '##원', '##구에서'], ['은', '##평', '##구에서'], ['서', '##대문', '##구에서'], ['마포', '##구에서'],
                  ['양', '##천', '##구에서'], ['강', '##서', '##구에서'], ['구', '##로', '##구에서'], ['금', '##천', '##구에서'],
                  ['영', '##등', '##포', '##구에서'], ['동작', '##구에서'], ['관', '##악', '##구에서'], ['서초', '##구에서'],
                  ['송', '##파', '##구에서'], ['강', '##동', '##구에서'], ['대', '##면으로'], ['온라인으로']],
        'CONTENT': [['교육'], ['전시'], ['산', '##림'], ['여가'], ['공', '##원'], ['문화'], ['행사'], ['역사'], ['전시', '##회가'], ['박', '##물', '##관에']],
        'STATUS': [['예약'], ['진행'], ['종료'], ['끝'], ['시작'], ['취소'], ['등록'], ['참여'], ['열', '##리는']],
        'INFO': [['어디서'], ['어디에'], ['링크'], ['어떻게'], ['정보가'], ['전화']]}]
}