from joblib import load
import pandas as pd

data = pd.read_csv('./Hamburger.csv')

model_file = './YSY_model/decision_tree_03.pkl'

model = load(model_file)

print(model)

print('어서오세여 KDT 햄버거 집입니다.')
print('당신이 먹을 햄버거의 영양성분이 어땠으면 좋겠나요?')
user_sodium = int(input('나트륨 : '))
user_sugar = int(input('설탕 : '))
user_fat = int(input('지방 : '))
user_protein = int(input('단백질 : '))

list = []
list.append(user_sodium)
list.append(user_sugar)
list.append(user_fat)
list.append(user_protein)

user_hamburger = int(model.predict([list])[0])

print(f"당신이 입력한 영양성분에 따라 먹을 햄버거의 칼로리는 {user_hamburger}kcal 입니다")

hbg_brand = []
hbg_name = []
hbg_cal_list = []
for i in range(data.shape[0]):
    if user_hamburger - 5 <= data.iloc[i][2] <= user_hamburger + 5:
        hbg_brand.append(data.iloc[i][0])
        hbg_name.append(data.iloc[i][1])
        hbg_cal_list.append(data.iloc[i][2])
print('다음과 같은 메뉴를 추천합니다 (오차 5kcal)')
for i in range(len(hbg_name)):
    print(f"[{i+1}] 브랜드 : {hbg_brand[i]}    메뉴 이름 : {hbg_name[i]}    칼로리 : {hbg_cal_list[i]}")
print('============= 이상 추천은 종료되었습니다. 맛있는 식사 되세요 :) ================')