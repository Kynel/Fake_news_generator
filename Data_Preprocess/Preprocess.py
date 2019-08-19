# 데이터를 모델에 넣을 수 있도록 가공하는 파일
# Written by RE-A
import os
import pandas as pd
import pickle

# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR,'Data_Preprocess','Data','Crawldata.xlsx')
data = pd.read_excel(DATA_DIR, sheet_name='Sheet1')
title_list = data['text']


def test(input_num):
    test_list = title_list[:10]
    sources_test = []
    targets_test = []

    for result in test_list:
        result = result.split(' ')
        print(result)
        try:
            source = result[:input_num]
            target = result[input_num:]
        except:
            print("파싱 오류. 제목의 단어 수가 3보다 작거나, 파싱할 수 없는 내용")
            break
        sources_test.append(source)
        targets_test.append(target)
        print("나눠짐")
        print(source)
        print(target)
        print('')
    print("나눈 것을 모은 리스트")
    print(sources_test)
    print('')
    print(targets_test)


def run(input_num):
    sources_stream = open('sources'+str(input_num)+'.dat', 'wb')
    targets_stream = open('targets'+str(input_num)+'.dat', 'wb')

    sources = []
    targets = []
    idx = 1

    for title in title_list:
        title = title.split(' ')
        try:
            source = title[:input_num]
            target = title[input_num:]
        except:
            print("파싱 오류. 제목의 단어 수가 너무 작거나, 파싱할 수 없는 내용 index:" + str(idx))
            break
        sources.append(source)
        targets.append(target)

    pickle.dump(sources, sources_stream)
    pickle.dump(targets, targets_stream)
    sources_stream.close()
    targets_stream.close()
    print("전처리 정상 종료")

# For test.
# test()
# For run
# run의 파라미터로 넣은 개수가 바로 source의 갯수이다.
# 예를 들어, 3을 넣으면 문장을 띄어쓰기 기준으로 3개 / 나머지 로 나눠서 저장시킨다.

print("source로 넣을 인풋의 갯수(Dimension) 입력. 입력한 갯수만큼의 첫 단어들/나머지 단어들로 분할된 리스트를 출력하여 저장합니다.\n")
input_value = input()
run(input_value)













