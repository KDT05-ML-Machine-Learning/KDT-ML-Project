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

| 패키지 이름  | 버전   | 사용 커맨드(Version command) |
| ------------ | ------ | ---------------------------- |
| Python       | 3.8.18 | python --version             |
| jupyter      | 1.0.0  | pip show jupyter             |
| ipython      | 8.12.2 | pip show ipython             |
| notebook     | 7.0.6  | pip show notebook            |
| numpy        | 1.24.3 | pip show numpy               |
| pandas       | 2.0.3  | pip show pandas              |
| matplotlib   | 3.7.2  | pip show matplotlib          |
| statsmodels  | 0.14.0 | pip show statsmodels         |
| skicit-learn | 1.3.0  | print(sklearn.**version**)   |

<hr/>

### KDT(Korea Digital Training)-ML(Machine Learning)

<hr/>

#### 사용한 데이터 사이트(수정 전)

1. [맥도날드](https://www.kaggle.com/datasets/mcdonalds/nutrition-facts)
2. [롯데리아](https://www.lotteeatz.com/upload/stg/etc/ria/items.html)
3. [미국 전 지점](https://www.kaggle.com/datasets/ulrikthygepedersen/fastfood-nutrition)
4. [버거킹](https://emilysinglelife.tistory.com/62)
5. [맘스터치](https://www.momstouch.co.kr/m/brand/notice-view.php?idx=49)
6. [노브랜드](https://realjace.tistory.com/entry/%EB%85%B8%EB%B8%8C%EB%9E%9C%EB%93%9C%EB%B2%84%EA%B1%B0-%EB%A9%94%EB%89%B4%EB%B3%84-%EC%B9%BC%EB%A1%9C%EB%A6%AC-%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84-%EC%B4%9D%EC%A0%95%EB%A6%AC)  
   7.[프랭크 버거](https://rooftoper.tistory.com/entry/%ED%94%84%EB%9E%AD%ED%81%AC-%EB%B2%84%EA%B1%B0-%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84)

<hr/>

###### 주제 : 야구 통계 분석

- 목차

* 1. 주제 선정 배경
* 2. 전처리
* 3. 모델 분석(명노아 : Lasso, Ridge, SVM)
* 4. 모델 분석(이승민 : KNR, LR)
* 5. 모델 분석(임소영 : DT, Boosting)
* 6. 모델 분석(박희진 : Bagging, Voting)
* 7. 최종 산출물 시연
  </hr>

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
    명노아
  </summary>

# 0. 데이터 크롤링()

### 관련 파일 : 명노아/01_Crawling.ipynb

사용 데이터셋 :  
[맥도날드](https://www.kaggle.com/datasets/mcdonalds/nutrition-facts)  
[롯데리아](https://www.lotteeatz.com/upload/stg/etc/ria/items.html)  
[미국 전 지점](https://www.kaggle.com/datasets/ulrikthygepedersen/fastfood-nutrition)  
[버거킹](https://emilysinglelife.tistory.com/62)  
[맘스터치](https://www.momstouch.co.kr/m/brand/notice-view.php?idx=49)  
[노브랜드](https://realjace.tistory.com/entry/%EB%85%B8%EB%B8%8C%EB%9E%9C%EB%93%9C%EB%B2%84%EA%B1%B0-%EB%A9%94%EB%89%B4%EB%B3%84-%EC%B9%BC%EB%A1%9C%EB%A6%AC-%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84-%EC%B4%9D%EC%A0%95%EB%A6%AC)  
[프랭크 버거](https://rooftoper.tistory.com/entry/%ED%94%84%EB%9E%AD%ED%81%AC-%EB%B2%84%EA%B1%B0-%EC%98%81%EC%96%91%EC%84%B1%EB%B6%84)

사용 모듈 : BeautifulSoup, urllib, pytesseract

<hr/>

# 1. 데이터 전처리

### 관련 파일 : 명노아/02_Preprocessing.ipynb

- 공통 feature, target 설정
- feature : 소금, 설탕, 지방, 단백질
- target : 칼로리

애로사항 : 맘스터치의 경우, 이미지 데이터로 있었는데 수작업이 여전히 필요하여 데이터셋에 합치지 않음
(moms_touch.png, moms_touch.txt)

<hr/>

# 2. 분석 파이프라인

- 1.  all_estimator로 회귀 모델 갖고오기
- 2.  Ridge, Lasso, SVM 모델 선별
- 3.  전처리 데이터 로드, 데이터 셋 분할(훈련용, 시험용)
- 4.  데이터 정규화
- 5.  이상값 제거(Z-score)
- 6.  데이터 분포, 상관계수 파악
- 7.  하이퍼 파라미터를 제외한 최적의 random_state 파악
- 8.  GridsearchCV를 사용하여 하이퍼 파라미터 찾기
- 9.  모델 데이터 파일 저장 (명노아/model)
- 10. 최종 산출물에 모델 파일 반영
  <hr/>

# 3. Ridge, Lassso를 사용한 모델 분석

### 관련 파일 :

명노아/03_Ridge.ipynb,  
04_Lasso.ipynb,  
05_SVM.ipynb,  
06_SVM Visual.ipynb

사용한 머신러닝 모델

1. Ridge
2. KernelRidge
3. RidgeCV
4. BayesianRidge
5. Lasso
6. LassoCV
7. LassoLars
8. LassoLarsCV
9. SVR
10. <hr/>

# 4. 데이터 분석 결과 시각화

- 1. SVM 분류 기법으로 보았을 때, 일반적으로 햄버거 데이터가 제일 많겠지만 음료수나 감자튀김 추가 메뉴나 세트메뉴 등등...  
     => 이상치 값이 너무 많기에, 제대로 된 분류가 되지 않는 모습  
     ![alt text](./명노아/img/image.png)
- 2. 왜 SVR 모델 중에서, LinearSVR은 높게 나왔는가?
     => 다차원 매핑을 시도하는 SVR, NuSVR과는 달리,직선을 긋는 LinearSVR은 상관관계가 높은 지방, 단백질에 초점을 잡아 정확률이 높게 나옴  
     ![alt text](./명노아/img/image-2.png)
- 3. 전체 모델 성능 시각화
  => 01. KernelRidge : 다차원 공간에 매핑하면서도, 특정 계수의 가중치(계수)를 0으로 낮춤으로써 높은 정확률을 보임 : 비선형 관계를 갖는 데이터에 적합하다  
  => 02. LassoLars : 데이터의 일부분만 사용하고, 계수의 축소 경로를 제공하므로, 정규화 강도를 조절하면서 모델을 세밀하게 튜닝할 수 있다  
  ![alt text](./명노아/img/image-1.png)
  <hr/>

# 5. 결론

- KernelRidge를 제일 성능이 좋은 모델로 확정
![alt text](./명노아/img/image-3.png)
<hr/>

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
    박희진(앙상블 - 배깅(RandomForest) / 보팅(Voting)
  </summary>

</details>
dsjfsjdfds
<hr/>
