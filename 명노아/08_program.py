from joblib import load
import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.preprocessing import StandardScaler
from IPython.display import display


# Scaler fitting
df=pd.read_csv('./Hamburger.csv')
scaler=StandardScaler()
scaler.fit(df[["sodium", "sugar", "total_fat", "protein"]])

ending='''
                       . . .                      
                ..:iiiiiiiiiiiii:..               
              .iii:::::::::::::::iiri:            
            .ir:::::::::::::::::::::iir.          
           :ii::::::::::::::::::.::::::i:         
          ii:::.. ...::::::::::.. ..::::i:        
         i:::::..1XJ..::::::::..sgP:.::::i.       
        i::.::..BBBBB.::::::::.RBBBB..:.::i       
      .r7r::.:.iBBBBB.::::::.:.bBBBd .:iirr.      
    :7vv7777r...rPEu:..::::::::.ir: :r7r77Lv.     
   .vii7rrrrrr.......:::::::::::.  Lii:iirr77.    
   :7:irrri:KBs :::.:::::::::.:.: 2Qg.7Yiirr7:    
   :rii7rii:bBD  :::::::.::::::.. XBBBBBirr77:    
    vrirrvBBBQBS: ....::.:::... :QBB1XSiirvv7     
     ii7775bJ:2BBBv.   . ..  .iZBBX..:irr7v:      
        ii:... .UBBBBg5vvv1XQBBQE: ...::iii       
         :::::.. .:sEBBBBBQBBd7. ..:::...i.       
          ii:::::...  ..... . ....::::::ii        
           iii::::::.......:.:.::::::::ii.        
            .ii:::::::.:::::::::::::iir:          
              .iii:i:i:::::::i:i:iir:.            
                ..::i:iiiiiii:::...               
 '''

hamburger='''   
              << 햄버거 먹고 싶니?? >>      
    ...    ...   ...    ...    ...   ...    ...   
.                                                 
. ..   ...    ...    ..    ...    ...    ..    .. 
                                .     .     .     
    ...    ...                       ...    ...   
. .     .           :r1SqSK5jr.           .     . 
. ..   ...      iKgBBB2Lru7jKQBBQJ.      ..    .. 
    . .       rBBKXjr.   i      rDBB.       . .   
    ...      BBr v.   .s  .j.      UBS      ...   
. .     .   BB :.                    QK   .     . 
. .    ...  B7                       PB  ...   .. 
    . .     UBQMBQBQBdQBBQBQMEBBBQBMBBr     . .   
     .      BJ     vBu. .:. :dR:     RB     ..    
. .     ..  LM112JvidBB7  :XBBjiLsUu1Qi  ..    .. 
         .   rBBBBQBBBBBQBQBBBBBBBQBQ.   .     .. 
    ...     BM. ..:.. ..rv:.  r:.i::rQg     ...   
            QS                r:.7iiiBB           
. .    ...  .BBI2YYv12IuYLYuPgX251KPBB   ..    .. 
         .    i1u212u2uUUUU1sYvLvLvv.             
    ...    ..                         ..    ...   
                                                  
. .    ...    ...    ..    ...    ...    ..    .. 
                                                  
    ...    ..    ...    ...    ..    ...    ...   
'''

model_list=["BayesianRidge","KernelRidge","Lasso","LassoCV","LassoLars","Ridge","RidgeCV","KNeighborsRegressor","LinearRegression","Boosting","Decision_tree","RandomForest","Voting"]
model_text=['''**BayesianRidge:**

**설명:** BayesianRidge는 베이지안 회귀 모델로, 가중치에 대한 사전 분포를 정의하고 베이지안 추론을 사용하여 가중치를 조절하는 회귀 모델입니다. 오차 항과 가중치에 대한 확률 분포를 고려하여 모델링됩니다.

**하이퍼파라미터:** alpha_1, alpha_2, lambda_1, lambda_2, alpha_init, lambda_init 등.
''',
'''
**KernelRidge:**

**설명:** KernelRidge는 커널 트릭을 사용하는 Ridge 회귀의 확장입니다. 비선형 데이터를 모델링할 수 있도록 다양한 커널 함수를 사용할 수 있습니다.

**하이퍼파라미터:** alpha, kernel, gamma 등.''',
'''
**Lasso:**

**설명:** Lasso는 L1 규제를 사용하는 선형 회귀 모델로, 특정 특징들의 가중치를 0으로 만들어 특징 선택에 활용됩니다.

**하이퍼파라미터:** alpha (규제 강도).''',
'''
**LassoCV:**

**설명:** LassoCV는 교차 검증을 사용하여 최적의 alpha 값을 자동으로 찾아주는 Lasso 모델입니다.

**하이퍼파라미터:** eps, n_alphas, cv 등''',
'''
**LassoLars:**

**설명:** LassoLars는 Least Angle Regression (LARS) 알고리즘을 사용하는 Lasso 모델로, 계수의 추정치를 조절하면서 변수를 선택할 수 있습니다.

**하이퍼파라미터:** alpha (규제 강도).''',
'''
**Ridge:**

**설명:** Ridge는 L2 규제를 사용하여 선형 회귀 모델을 구축합니다. Lasso와 달리 계수를 0에 가깝게 만들지 않고, 작은 값을 유지합니다.

**하이퍼파라미터:** alpha (규제 강도)..''',
'''
**RidgeCV:**

**설명:** RidgeCV는 교차 검증을 사용하여 최적의 alpha 값을 자동으로 찾아주는 Ridge 모델입니다.

**하이퍼파라미터:** store_cv_values, alphas 등.''',
'''
**KNR (K-neighbors regression):**

**설명:** KNR은 K-최근접 이웃 알고리즘을 사용한 회귀 모델로, 주어진 데이터 포인트에 가장 가까운 K개의 이웃의 평균값이나 가중 평균값을 사용하여 예측을 수행합니다.

**하이퍼파라미터:** n_neighbors, weights, algorithm 등.
''',
'''
**LR (Linear Regression):**

**설명:** Linear Regression은 선형 모델로, 입력 특징과 가중치의 선형 조합으로 예측을 수행하는 회귀 알고리즘입니다. 가장 간단하면서도 효과적인 회귀 방법 중 하나입니다.

**하이퍼파라미터:** 없음 (주로 최소제곱법을 사용하며, 규제를 위한 하이퍼파라미터가 없을 수 있음).
''',
'''
**Boosting:**

**설명:** Boosting은 약한 학습자(weak learner)들을 결합하여 강력한 앙상블 모델을 만드는 알고리즘입니다. 이전 학습자의 오차에 가중치를 부여하면서 순차적으로 학습을 진행하여 모델의 성능을 향상시킵니다.

**하이퍼파라미터:** n_estimators, learning_rate, max_depth 등.
''',
'''
**Decision_tree:**

**설명:** Decision Tree는 데이터를 분할하고 각 분할에서 예측을 수행하는 트리 구조의 모델입니다. 각 분할은 특정 조건을 기반으로 결정되며, 데이터를 계층적으로 분류하여 의사 결정을 수행합니다.

**하이퍼파라미터:** max_depth, min_samples_split, min_samples_leaf 등.
''',
'''
**Random Forest:**

**설명:** Random Forest는 여러 개의 의사 결정 트리를 구성하고 각 트리의 예측을 결합하여 더 강력하고 안정적인 모델을 형성하는 앙상블 학습 방법입니다. 각 트리는 부트스트랩 샘플링을 통해 데이터의 일부를 사용하며, 무작위로 선택된 특징들을 사용하여 높은 다양성을 유지합니다.

**하이퍼파라미터:** n_estimators, max_depth, min_samples_split 등.

''',
'''
**Voting:**

**설명:** Voting은 여러 개의 다른 머신러닝 모델을 조합하여 높은 성능의 앙상블 모델을 형성하는 앙상블 학습 방법 중 하나입니다. 여러 모델의 예측을 조합함으로써 개별 모델의 약점을 상쇄하고, 전체적인 성능을 향상시킬 수 있습니다. Voting은 주로 분류(Classification) 및 회귀(Regression) 문제에 사용됩니다.

**하드 보팅(Hard Voting):** 다수결 원칙을 적용하여 각 모델의 예측 중 가장 많이 선택된 클래스나 값으로 최종 예측을 수행합니다.

**소프트 보팅(Soft Voting):** 각 모델의 예측에 가중치를 부여하여 조합하며, 가중 평균을 계산하여 최종 예측을 수행합니다. 이는 모델의 예측에 대한 확률이나 신뢰도 정보를 활용하는 방식입니다.

**하이퍼파라미터:** Voting은 다양한 모델을 조합하기 때문에 각 모델의 하이퍼파라미터를 설정해야 하며, 가중치를 조절하는 등의 하이퍼파라미터도 있을 수 있습니다.
''']

for i in hamburger:
    if i =="\n":
        print()
    else :print(i, end="",sep="")
    
while True:
    while True:
        print("====================================================================================") # 모델 선택    
        print("종료키 : 0")

        print("사용할 모델을 선택하세요~")
        for i in range(len(model_list)):
            print(f"{i+1}. {model_list[i]}")
        print("====================================================================================")
        model_num=int(input("모델 번호를 입력하세요. : "))
        os.system('cls')
        if not model_num:
            break
        print("====================================================================================") # 모델 설명
        print(model_text[model_num-1])
        yesno=int(input("해당 모델을 사용하시겠습니까? (예:1/아니오:0) :"))
        os.system('cls')
        if yesno :
            break
        if not yesno:
            continue
    try :
        df=pd.read_csv('./Hamburger.csv')
    except Exception as e:
        print(e)
    print(f"{model_list[model_num-1]}모델 불러오기...(성공!)")
    
    model_file=f"./model/{model_list[model_num-1]}.pkl"
    try :
        model=load(model_file)
    except Exception as e:
        print(e)

    # 입력순서 : 소금, 설탕, 지방, 단백질
    # 섭취량 기준 : https://www.khidi.or.kr/kps/dhraStat/result5?menuId=MENU01657&siteId=SITE00002
    # 설탕 섭취량 : https://feelgoodpal.com/ko/blog/how-much-sugar-per-day/
    # =======================================================================
    #   소금                설탕                   지방         단백질        =
    #  3299.0        여자(25), 남자(37.5)         47.85       72.4(38.56)    = 
    # =======================================================================

    # 남자 기준 섭취량 : 2500
    # 여자 기분 섭취량 : 1800

    # 한 끼 식사량 : 1/3
    # 한 끼 칼로리 섭취량(남자) : 2500/3
    # 한 끼 칼로리 섭취량(여자) : 1800/3
    # 감자튀김 평균 칼로리 : 250~300
    # 음료수 평균 칼로리 : 0~50

    # 적정 칼로리 섭취량(남자) : 2500/3 - (250~300) - (0~50) = 480~580Kcal 정도, 감자튀김을 포기하면 780~880Kcal 정도
    # 적정 칼로리 섭취량(여자) : 1800/3 - (250~300) - (0~50) = 250~350Kcal 정도, 감자튀김을 포기하면 550~650Kcal 정도 



    print("====================================================================================") # 입력파트 
    gender=int(input("성별을 입력하세요(남자:0/여자:1) : ")) # 성별에 따라 일일섭취량이 정해져있으므로, 성별에 따라 섭취량을 다르게 설정
    print("====================================================================================")
    if gender in [0,1]:
        sugar=int(input("얼마나 달달한게 땡기시나요?(1~10) :  "))
        if gender :
            sugar=sugar*25/10
        else :
            sugar=sugar*37.5/10              # 1. 설탕 입력
    else :
        print("잘못된 입력입니다.")
        sugar=15
        
    # 2. 소금 입력
    salt=int(input("얼마나 짭짤한게 떙기시나요?(1~10) : "))
    salt=salt*3299/10

    # 3. 지방 입력
    fat=int(input("얼마나 기름진 게 땡기시나요?(1~10) : "))
    fat=fat*47.85/10

    # 4. 단백질 입력
    protein=int(input("얼마나 단백질이 땡기시나요?(1~10) : "))
    protein=protein*38.56/10
    os.system('cls')
    print("====================================================================================")
    if model_num <10 : 
        data=pd.DataFrame([[salt, sugar, fat, protein]], columns=["sodium", "sugar", "total_fat", "protein"])
    elif model_num >=10 :
        data=pd.DataFrame(scaler.transform([[salt, sugar, fat, protein]]), columns=["sodium", "sugar", "total_fat", "protein"])


    cal=model.predict(data)
    res=df[(df["calories"] <= cal[0]+10) & (df["calories"]>=cal[0]-10)]
    res.sort_values(by="restaurant", ascending=True).reset_index(drop=True)
    print(f"총 {res.shape[0]}건의 음식이 검색되었습니다! ")


    print("====================================================================================") # 정렬파트 
    # 정렬방식 선택

    print("무엇을 기준으로 정렬하시겠습니까? ( 0입력 시 종료 )")
    
    sort_num = (int(input("1. 짭짤함 \n2. 달달함 \n3. 기름짐 \n4. 단백질 : ")))
    if not sort_num:
        break
    os.system('cls')
    res=res.sort_values(by=["sodium","sugar","total_fat","protein"][sort_num-1], ascending=False).reset_index(drop=True)
    print(f"{'===='*40}") 
    print(f"{'Restaurant':^15} {'Item':^100} {'Calories':^10}")
    for index, row in res[["restaurant","item","calories"]].iterrows():
        restaurant = row['restaurant'].strip()
        item = row['item'].strip()
        calories = row['calories']
        
        print(f"{restaurant:^15} {item:^100} {calories:10}")
    print(f"{'===='*40}\n\n         <<< 맛있게 드세요! >>> \n\n\n")  
    for i in ending:
        if i =="\n":
            print()
        else :print(i, end="",sep="")
    
    if not int(input("한번 더 할까요?(네:1,아니오:0) : ")) : 
        # 프로그램 종료
        os.sys.exit(1)
    os.system('cls')