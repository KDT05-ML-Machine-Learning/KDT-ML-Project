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
    이승민(소주제)
  </summary>

</details>

</hr>

<details>
  <summary>
    박희진( 앙상블 - 배깅(RandomForest) / 보팅 )
  </summary>
  
## (1) 모델 선정 이유
  
- 과대적합때문에 늘 고생했던 경험이 있어서 과대적합을 완화해주는데 알맞은 RandomForest 모델을 선정하였다.
- 우리 프로젝트가 같은 데이터로 다른 모델의 성능을 비교 파악 하는 것이기 때문에 보팅의 알고리즘과 매우 유사하다고 느꼈고, 우리가 비교해서 가장 높은 성능을 가진 모델의 결과와 보팅모델의 결과를 비교해보면 재밌겠다는 생각이 들어 선정하였다.
  
## (2) 데이터 파악 및 전처리
  
- data : restrant, item, sodium, sugar, total_fat, portein, caloriest
- target : caloriest
- feature : item, sodium, sugar, total_fat, portein
- restrant는 순서가 없는 범주형 데이터 -> OneHotEncoding 실시
  - 같은 브랜드지만 다른 이름인 데이터가 있길래 통일
  - targer과 상관계수 파악 -> 큰 상관관계 파악 X -> 무시 
- 결측치 제거
    - 대체했을 때, 데이터가 왜곡될까봐 대체하지 않고 제거함.
- 중복치 제거
- 이상치 확인
  ![image](https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/07ef4a97-16b8-4759-a0f9-54c9aa671ba9)
    - 이상치가 매우 많이 확인 되었으나 잘못 입력된 데이터가 아니라는 판단하에 제거하지 않고 진행
    - 이상치에 영향을 덜 받는 MinMaxScaler나 RobustScaler 사용하는 것이 좋겠다.
- feature data를 산점도 찍어 봤을때, 선형 또는 묘하게 2차 곡선을 띰
  ![image](https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/253fb431-be29-4f03-9d4a-e1ad6310f1fc)
  - feature들끼리 상관관계 있는지 파악
    - total_fat과 sodium 상관관계 높음
    - total_fat과 sodium feature만 poly 진행하여 모델 학습해봤으나 과대적합되어 기각
- MinMaxScaler 적용하여 스케일링
  - 세 방법중에 MAE와 RMSE가 제일 낮은 Scaler 선택
- train_test_split 메서드의 최적의 random_state 값 찾기
- RandomForest 메서드의 최적의 random_state 값 찾기
  
  
## (3) 모델 학습 및 모델 평가  
  
### __[RandomForest model]__  
  
- train_score : 0.98 / test_score : 0.95 -> 과대적합이라고 판단
  -  과대적합을 방지하기 위해 튜닝 진행
    - n_estimators, max_depth, min_samples_split, max_features 파라미터 튜닝
  - 과대적합을 방지하기 위해 교차검증 진행
    - GridSearchCV를 통해 최적의 모델 산출
** 과대 적합 해결! ** 
- 튜닝 후 : train_score : 0.96 / test_score : 0.95
  
<aside>
💡 최적의 모델  
  
  ![image](https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/00edaf5c-124d-4241-99e0-dd3c784497f0)  
      
  <img  width="300" height="200" alt="image" src="https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/04d8f60f-769c-43d4-83a7-f7635ce3922b">

</aside>
  
### __[Voting model]__  
  
- 각 조원들과 내가 만들었던 최적의 모델을 estimators 파라미터 안에 넣어 모델 생성 및 학습
- train_score : 0.91 / test_score : 0.91 -> 최적적합
  
<aside>
💡 최적의 모델  

  ![image](https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/b8271ef0-88d3-4e0d-b8e2-432e449c059f)  
              
  <img  width="300" height="200" alt="image" src="https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/39c226e7-6d58-4c61-8c73-1bd63b4027d4">

</aside>
  
## (4) 새로운 데이터로 칼로리 예측  
- 맘스터치 화이트갈릭싸이버거의 나트륨, 당류, 포화지방, 단백질 데이터를 model에 넣어 predict하여 값 예측  
- 각 모델 별로 예측값과 오차 도출
  ![image](https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/5010f150-41a6-4900-82c9-1f8f44c37450)
  
## (5) 결과  
  
![image](https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/6512e391-14f7-476c-affc-19b7ffad4cf1)
![image](https://github.com/ParkHeeJin00/KDT-5_MLProject/assets/155441547/d43885dc-5429-4488-b2e4-991833ecc75d)
- boost model이 score가 가장 높고, 최적적합에다, 새로운 데이터를 넣었을때도 MAE와 RMSE값이 낮다.  
  
## (6) 활용
- 칼로리 예측을 기반한 햄버거 추천 프로그램  
  
## (7) 피드백  
  
- 이상치가 많은 feature data에서 MinMaxScaler를 잘 사용하였다.
  - 이상치 제거를 안해도 MinMaxScaler로도 어느정도 이상치 정리가 된다.  

</details>
<hr/>

###### 출처/데이터

| 관련 자료명 | URL |
| :---------: | --- |
