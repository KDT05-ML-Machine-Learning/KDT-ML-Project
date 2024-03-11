# _KDT05-Machine Learning Project_

경북대학교 KDT(Korea Digital Training) 빅데이터 전문가 양성과정 5기 : ML(Machine Learning) 3팀입니다

임소영 : [깃허브 링크](https://github.com/YimSoYoung1001)  
박희진 : [깃허브 링크](https://github.com/ParkHeeJin00)  
이승민 : [깃허브 링크](https://github.com/winmin94)  
명노아 : [깃허브 링크](https://github.com/noah2397)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)  
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

<hr/>

#### 개발환경

| 패키지 이름 | 버전   | 사용 커맨드(Version command) |
| ----------- | ------ | ---------------------------- |
| Python      | 3.8.18 | python --version             |
| jupyter     | 1.0.0  | pip show jupyter             |
| ipython     | 8.12.2 | pip show ipython             |
| notebook    | 7.0.6  | pip show notebook            |
| numpy       | 1.24.3 | pip show numpy               |
| pandas      | 2.0.3  | pip show pandas              |
| matplotlib  | 3.7.2  | pip show matplotlib          |
| statsmodels | 0.14.0 | pip show statsmodels         |

<hr/>

### KDT(Korea Digital Training)-ML(Machine Learning)

<hr/>

#### 사용한 데이터 사이트(수정 전)

1. [KBO 홈페이지](https://www.koreabaseball.com/Default.aspx)
2. [Daum 스포츠-야구](https://sports.daum.net/record/kbo/team?season=2023)
3. [구글공유 PPT](https://docs.google.com/presentation/d/1iw8iwN1F_FjeJlKNg46WBwOhtqjZGTJt9zUaESa8WAY/edit)

<hr/>

###### 주제 : 야구 통계 분석

- 목차

* 1. 배경
* 2. (임소영)
* 3. (박희진)
* 4. (명노아)
* 5. (이승민)

###### 역할 분담

|          역할 | 참여인원                       |
| ------------: | ------------------------------ |
|      주제선정 | 임소영, 박희진, 명노아, 이승민 |
|          코딩 | 임소영, 박희진, 명노아, 이승민 |
|          발표 | 임소영, 박희진, 명노아, 이승민 |
|       git관리 | 임소영, 박희진, 명노아, 이승민 |
|   Readme 작성 | 임소영, 박희진, 명노아, 이승민 |
|      PPT 제작 | 임소영, 박희진, 명노아, 이승민 |
| PPT 관리,병합 | 임소영, 박희진, 명노아, 이승민 |

### 소주제 개요(개인 항목)

<details>
  <summary>
    임소영(소주제)
  </summary>
</details>

</hr>

<details>
  <summary>
    명노아(소주제)
  </summary>

</details>

</hr>

<details>
  <summary>
    이승민(KNR(K-최근접 이웃 회귀), LR(선형 회귀))
  </summary>
K-Nearest Neighbors Regression
  1. 데이터 전처리 실시
  - 필요한 부분으로 데이터를 분할해서 저장
  - 이상치도 필요한 내용을 담고 있어 제거하지 않음
  2. 데이터셋 준비
  - 최적의 random_state 추적
  3. 학습 및 평가
  - KNR : 과대적합이 발생하여 튜닝 진행
  - [튜닝 1] K값 조절
  - [튜닝 2] 가중치 조정
  4. 예측값 구하기 및 성능 평가
  - 튜닝을 실시한 2가지 모델에 대해 예측값을 구하고 성능을 평가함
  - 성능 평가 요소(R2 score, MAE, MSE)
  5. 모델 저장(.pkl 형식)
  - 둘 중 성능이 좋은 '튜닝2' 모델을 최종 모델로 저장

Linear Regression
  1. 데이터 전처리 실시
  - 필요한 부분으로 데이터를 분할해서 저장
  - 이상치로 필요한 내용을 담고 있어 제거하지 않음
  2. 데이터셋 준비
  - 최적의 random_state 추적
  3. 학습 및 평가
  - LR : 차이는 적으나 좀 더 스코어를 끌어올릴 수 있을 거라 예상되어 튜닝 진행
  - [튜닝] fit-intercept 조정
  4. 예측값 구하기 및 성능 평가
  - 튜닝을 실시한 모델에 대해 예측값을 구하고 성능을 평가
  - 성능 평가 요소(R2 score, MAE, MSE)
  5. 모델 저장(.pkl 형식)
</details>

</hr>

<details>
  <summary>
    박희진(소주제)
  </summary>

</details>
<hr/>

###### 출처/데이터

| 관련 자료명 | URL |
| :---------: | --- |
