import pandas as pd
from itertools import product
import os

# 각 리스트와 패턴을 매칭하여 질문 생성
questions = []


# 시간 문장 생성
varT = {
    'TIME_sub': [" 중순", " 초", " 말", " 첫번째 주", " 두번째 주", " 세번째 주", " 네번째 주", " 마지막 주"],  # 못씀
    'TIME_0_q12': ["봄", "여름", "가을", "겨울", "여름 방학", "겨울 방학"],
    'TIME_1_q12': ["이번 달", "이달", "이번 주", "이주", "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월", "오전", "오후", "휴일"],
    'TIME_2_q12': ["주말", "평일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"],
    'TIME_3_q345': ["오늘", "요즘", "주말마다"],
    'TIME_4_q6': ["언제", "몇시에"],
    'TIME_5_q7': ["기간", "시간"]
        }
'''
1: "에 참여할 수 있는 프로그램은 무엇인가요?"
2: "에 참여 가능한 프로그램이 있나요?"

3: " 참여 가능한 프로그램이 있나요?"
4: " 인기 있는 프로그램은 어떤게 있나요?"
5: " 열리는 프로그램이 있나요?"

6: "이 프로그램은" "열리나요?"
7: "이 프로그램은이 열리는" "이 어떻게 되나요?"
'''
TIME_list = ["TIME_0_q12", "TIME_1_q12", "TIME_2_q12", "TIME_3_q345", "TIME_4_q6", "TIME_5_q7"]

for lst in TIME_list:
    if lst.split('_')[-1] == "q12":
        for s in varT[lst]:
            questions.append(s + "에 참여할 수 있는 프로그램은 무엇인가요?")
            questions.append(s + "에 참여 가능한 프로그램이 있나요?")

    elif lst.split('_')[-1] == "q345":
        for s in varT[lst]:
            questions.append(s + " 참여 가능한 프로그램이 있나요?")
            questions.append(s + " 인기 있는 프로그램은 어떤게 있나요?")
            questions.append(s + " 열리는 프로그램이 있나요?")

    elif lst.split('_')[-1] == "q6":
        for s in varT[lst]:
            questions.append("이 프로그램은 " + s + " 열리나요?")

    elif lst.split('_')[-1] == "q7":
        for s in varT[lst]:
            questions.append("이 프로그램이 열리는 " + s + "이 어떻게 되나요?")
            questions.append("이 프로그램은 열리는 " + s + "이 어떻게 되나요?")


varC = {
    'TARGET_0_q135': ["청소년", "성인", "가족", "어른", "초등학생", "중학생", "고등학생", "대학생", "장애인", "어르신", "다문화 가정", "청년"],
    'TARGET_1_q236': ["유아", "어린이", "단체", "부모와 아이"],
    'TARGET_2_q346': ["가족 단위"]
        }
'''
1: "이 참여할 수 있는 프로그램은 무엇인가요?"
2: "가 참여할 수 있는 프로그램은 무엇인가요?"
3: "만 참여할 수 있는 프로그램은 무엇인가요?"
4: "로 참여할 수 있는 프로그램은 무엇인가요?"
5: "을 위한 프로그램 정보가 궁금해요"
6: "를 위한 프로그램 정보가 궁금해요"
'''
TARGET_list = ['TARGET_0_q135', 'TARGET_1_q236', 'TARGET_2_q346']
for lst in TARGET_list:
    if lst.split('_')[-1] == "q135":
        for s in varC[lst]:
            questions.append(s + "이 참여할 수 있는 프로그램은 무엇인가요?")
            questions.append(s + "만 참여할 수 있는 프로그램은 무엇인가요?")
            questions.append(s + "을 위한 프로그램 정보가 궁금해요")

    elif lst.split('_')[-1] == "q236":
        for s in varC[lst]:
            questions.append(s + "가 참여할 수 있는 프로그램은 무엇인가요?")
            questions.append(s + "만 참여할 수 있는 프로그램은 무엇인가요?")
            questions.append(s + "를 위한 프로그램 정보가 궁금해요")

    elif lst.split('_')[-1] == "q346":
        for s in varC[lst]:
            questions.append(s + "만 참여할 수 있는 프로그램은 무엇인가요?")
            questions.append(s + "로 참여할 수 있는 프로그램은 무엇인가요?")
            questions.append(s + "를 위한 프로그램 정보가 궁금해요")


PRICE = ["무료", "유료"]
for pr in PRICE:
    questions.append(pr + " 프로그램이 있으면 알려주세요")
    questions.append(pr + "로 참여할 수 있는 전시회가 있나요?")

PLACE = ["야외", "실내", "실외", "박물관", "공원", "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]
for pl in PLACE:
    questions.append(pl+ "에서 열리는 프로그램이 있나요?")

CONTENT = ["교육", "전시", "산림", "여가", "공원", "문화", "행사", "역사"]
# " 관련 프로그램을 추천해 주세요."
# "형 프로그램을 추천해 주세요."
for ct in CONTENT:
    questions.append(ct + " 관련 프로그램을 추천해 주세요.")
    questions.append(ct + "형 프로그램을 추천해 주세요.")

ONOFF = ["대면", "온라인"]
# "으로 참여 가능한 프로그램이 있나요?"
for of in ONOFF:
    questions.append(of+"으로 참여 가능한 프로그램이 있나요?")

varS = {'STATUS_0_q1': ["예약", "진행"],
        'STATUS_1_q2': ["종료", "끝", "시작"]}

STATUS_list = ['STATUS_0_q1', 'STATUS_1_q2']

for lst in STATUS_list:
    if lst.split('_')[-1] == "q1":
        for s in varS[lst]:
            questions.append("프로그램은 " + s + "중 인가요?")
    elif lst.split('_')[-1] == "q2":
        for s in varS[lst]:
            questions.append("프로그램은 " + s + " 인가요?")


varE = {'ETC_0_q13': ["예약 링크"],
        'ETC_1_q24': ["취소 기간", "등록 기간", "예약 기간"],
        'ETC_2_q5': ["어디서 예약", "어디에 전화", "어디서 취소", "어디서 등록", "어떻게 예약",  "어떻게 취소",  "어떻게 등록"]}

'''
1 : "가 어떻게 되나요?"
2 : "이 어떻게 되나요?"
3 : "가 무엇 인가요?"
4 : "을 어디서 확인하나요?"
5 : "하나요?"
'''

ETC_list = ['ETC_0_q13', 'ETC_1_q24', 'ETC_2_q5']

for lst in ETC_list:
    if lst.split('_')[-1] == "q13":
        for s in varE[lst]:
            questions.append(s+"가 어떻게 되나요?")
            questions.append(s+"가 무엇 인가요?")

    elif lst.split('_')[-1] == "q24":
        for s in varE [lst]:
            questions.append(s+"이 어떻게 되나요?")
            questions.append(s+"을 어디서 확인하나요?")

    elif lst.split('_')[-1] == "q5":
        for s in varE[lst]:
            questions.append("프로그램은 " + s + " 하나요?")

df = pd.DataFrame(questions, columns=["questions"])
path = os.path.join(os.getcwd(), "DataNer/MakeQuestionNer.csv")
df.to_csv(path, index=False)




