# Fake_news_generator
Make "Fake news generator" for educational purpose. with RNN

### 프로젝트 설명

- 현재는 가짜 뉴스 제목(Title) 만 자동으로 생성하는 걸 목표로 함.
- 사용자가 어떤 단어든 입력한다면, 모델에 그에 맞춰 그럴듯한 가짜 뉴스 제목을 생성해 주는 방식.
- 정치/세계 분야에서 추출한 데이터를 토대로 학습이 이루어질 예정.


### 사용한 라이브러리 및 코드

Google Tensorflow / Keras

https://github.com/jason9693/NNST-Naver-News-for-Standard-and-Technology-Database

https://github.com/deeplearningzerotoall


### 뉴스 제목 생성기 사용법(임시)

기본적으로 model.py에서 학습을 진행하고, 체크포인트를 만들고
model_test.py에서 체크포인트를 불러와 사용하는 구조이다.

model_test.py를 실행한후, 콘솔에서 sentence의 값을 변경하고 test(...) 함수를 실행하는것으로 간단한 테스트가 가능하다.
(현재 epoch=100 으로 학습되어 있음)

model.py에서 학습완료된 모델에서 볼수있는 값과 model_test.py에 입력하여 볼 수 있는 값이 같은것으로 동일한 모델을 불러왔음을 알 수 있다.

test.py에 불필요한것들을 제거해야하는데 그건 
