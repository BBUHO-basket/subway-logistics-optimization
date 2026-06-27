import pandas as pd
from haversine import haversine
from pathlib import Path

BASE_DIR = Path(__file__).parent
data_warehouse =  pd.read_csv(BASE_DIR / "data/raw/warehouse_xy.csv")

data_sub = pd.read_csv(BASE_DIR / "data/raw/subway.csv")

data_park = pd.read_csv(BASE_DIR / "data/raw/data_car.csv")

print(data_warehouse.columns)
print(data_sub.columns)
print(data_park.columns)

total_warehouse = [] 

for i in range(0, len(data_sub)) :
    temp = []
    start = (data_sub.iloc[i,3], data_sub.iloc[i,4])
    
    for j in range(0, len(data_warehouse)) : 
        goal = (data_warehouse.iloc[j,3], data_warehouse.iloc[j,4])
        temp.append(haversine(start, goal))
    
    total_warehouse.append(temp)

k_temp = []
for i in total_warehouse :
    print(sum(i)/len(data_warehouse))
    k_temp.append(sum(i)/len(data_warehouse))
    

#%% 

total_park = [] 

for i in range(0, len(data_sub)) :
    temp = []
    start = (data_sub.iloc[i,3], data_sub.iloc[i,4])
    
    for j in range(0, len(data_park)) :
        goal = (data_park.iloc[j,4], data_park.iloc[j,5])
        temp.append(haversine(start, goal))
    temp.sort()
    print(temp[0:10])
    total_park.append(sum(temp[0:10])/10)


for i in total_park :
    print(i)

#%%
# SF 생성 및 sub_location.csv 업데이트

import numpy as np
from sklearn.preprocessing import StandardScaler

# 창고까지 거리 + 주차장까지 거리 합산 (짧을수록 물류 접근성이 좋음)
# 부호를 반전하여 짧은 거리 = 높은 점수로 변환 후 표준화
raw_sf = np.array(k_temp) + np.array(total_park)
sf_scaled = StandardScaler().fit_transform(-raw_sf.reshape(-1, 1)).flatten()

import os

loc_path = BASE_DIR / "data/processed/sub_location.csv"
tmp_path = BASE_DIR / "data/processed/sub_location_tmp.csv"

data_loc = pd.read_csv(loc_path, encoding='utf-8')
data_loc['SF'] = sf_scaled
data_loc.to_csv(tmp_path, index=False, encoding='utf-8-sig')
os.replace(tmp_path, loc_path)

print("SF 업데이트 완료")
print(data_loc[['name_subway', 'SF']].head(10).to_string())
