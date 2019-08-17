# 데이터를 모델에 넣을 수 있도록 가공하는 파일
# Written by RE-A
import os
import pandas as pd
import pickle

# Directories
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR,'Naver_News_Title_Crawler','data.xlsx')

data = pd.read_excel(DATA_DIR, sheet_name='Sheet1')
title_list = data['text']

def test():
    test_list = title_list[:10]
    sources_test = []
    targets_test = []

    for result in test_list:
        result = result.split(' ')
        print(result)
        try:
            source = result[:3]
            target = result[3:]
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


def run():
    sources_stream = open('sources.dat', 'wb')
    targets_stream = open('targets.dat', 'wb')

    sources = []
    targets = []
    idx = 1

    for title in title_list:
        title = title.split(' ')
        try:
            source = title[:3]
            target = title[3:]
        except:
            print("파싱 오류. 제목의 단어 수가 3보다 작거나, 파싱할 수 없는 내용 index:" + idx)
            break
        sources.append(source)
        targets.append(target)

    pickle.dump(sources, sources_stream)
    pickle.dump(targets, targets_stream)
    sources_stream.close()
    targets_stream.close()
    print("전처리 정상 종료")

# For test.
test()
# For run
# run()













