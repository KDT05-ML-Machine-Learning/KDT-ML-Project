from joblib import load
import pandas as pd
import numpy as np
import sklearn
import os

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

model_list=["BayesianRidge","KernelRidge","Lasso","LassoCV","LassoLars","Ridge","RidgeCV"]
model_text=['''BayesianRidge:

설명: BayesianRidge는 베이지안 회귀 모델로, 가중치에 대한 사전 분포를 정의하고 베이지안 추론을 사용하여 가중치를 조절하는 회귀 모델입니다. 오차 항과 가중치에 대한 확률 분포를 고려하여 모델링됩니다.
하이퍼파라미터: alpha_1, alpha_2, lambda_1, lambda_2, alpha_init, lambda_init 등.''',
'''
KernelRidge:

설명: KernelRidge는 커널 트릭을 사용하는 Ridge 회귀의 확장입니다. 비선형 데이터를 모델링할 수 있도록 다양한 커널 함수를 사용할 수 있습니다.
하이퍼파라미터: alpha, kernel, gamma 등.''',
'''
Lasso:

설명: Lasso는 L1 규제를 사용하는 선형 회귀 모델로, 특정 특징들의 가중치를 0으로 만들어 특징 선택에 활용됩니다.
하이퍼파라미터: alpha (규제 강도).''',
'''
LassoCV:

설명: LassoCV는 교차 검증을 사용하여 최적의 alpha 값을 자동으로 찾아주는 Lasso 모델입니다.
하이퍼파라미터: eps, n_alphas, cv 등.''',
'''
LassoLars:

설명: LassoLars는 Least Angle Regression (LARS) 알고리즘을 사용하는 Lasso 모델로, 계수의 추정치를 조절하면서 변수를 선택할 수 있습니다.
하이퍼파라미터: alpha (규제 강도).''',
'''
Ridge:

설명: Ridge는 L2 규제를 사용하여 선형 회귀 모델을 구축합니다. Lasso와 달리 계수를 0에 가깝게 만들지 않고, 작은 값을 유지합니다.
하이퍼파라미터: alpha (규제 강도).''',
'''
RidgeCV:

설명: RidgeCV는 교차 검증을 사용하여 최적의 alpha 값을 자동으로 찾아주는 Ridge 모델입니다.
하이퍼파라미터: store_cv_values, alphas 등.''']

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
    data=pd.DataFrame([[salt, sugar, fat, protein]], columns=["sodium", "sugar", "total_fat", "protein"])


    cal=model.predict(data)
    res=df[(df["calories"] <= cal[0]+10) & (df["calories"]>=cal[0]-10)]
    res.sort_values(by="restaurant", ascending=True).reset_index(drop=True)
    print(f"총 {res.shape[0]}건의 음식이 검색되었습니다! ")


    print("====================================================================================") # 정렬파트 
    # 정렬방식 선택
    while True:
        print("무엇을 기준으로 정렬하시겠습니까? ( 0입력 시 종료 )")
        
        sort_num = (int(input("1. 짭짤함 \n2. 달달함 \n3. 기름짐 \n4. 단백질 : ")))
        if not sort_num:
            break
        os.system('cls')
        res=res.sort_values(by=["sodium","sugar","total_fat","protein"][sort_num-1], ascending=False).reset_index(drop=True)
        print("====================================================================================") 
        print(res[["restaurant","item","calories"]])
        print("====================================================================================") 

    #statics=df[["sodium","sugar","total_fat","protein"]].describe()
    #print(statics)