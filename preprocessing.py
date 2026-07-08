import pandas as pd
from haversine import haversine
from pathlib import Path

BASE_DIR = Path(__file__).parent
data_warehouse =  pd.read_csv(BASE_DIR / "data/raw/warehouse_xy.csv")

data_sub = pd.read_csv(BASE_DIR / "data/raw/subway.csv")

data_park = pd.read_csv(BASE_DIR / "data/raw/data_car.csv")

total_warehouse = [] 

for i in range(0, len(data_sub)) :
    temp = []
    start = (data_sub.iloc[i,3], data_sub.iloc[i,4])
    #i = 0 : [37.514,127.033]
    
    for j in range(0, len(data_warehouse)) : 
        goal = (data_warehouse.iloc[j,3], data_warehouse.iloc[j,4])
        # j = 0 : [36.491,126.831]
        # j = 1 : [37.491,126.831]
        temp.append(haversine(start, goal))
        # temp = [haversine(i=0,j=0),haversine(i=0,j=1)....]
    
    total_warehouse.append(temp)
    """total_warehouse = [[haversine(i=0,j=0),haversine(i=0,j=1)....]] 
                            ... 강남구청과 모든 창고까지의 거리
                         [[haversine(i=1,j=0),haversine(i=1,j=1)....]] 
                            ... 강남역과 모든 칭고까지의 거리
                                            ...                     ]"""

k_temp = []
for i in total_warehouse :
    k_temp.append(sum(i)/len(data_warehouse))
# k_temp = [강남구청과 모든 창고까지의 평균거리, 강남역과 모든 창고까지의 평균거리]
    

#%% 

total_park = [] 

for i in range(0, len(data_sub)) :
    temp = []
    start = (data_sub.iloc[i,3], data_sub.iloc[i,4])
    
    for j in range(0, len(data_park)) :
        goal = (data_park.iloc[j,4], data_park.iloc[j,5])
        temp.append(haversine(start, goal))
        #temp = [강남구청과 첫번째 주차장까지의 거리, 강남구청과 두번째 주차장까지의 거리, ...]
    temp.sort()
    #temp를 오름차순으로 나열 
    total_park.append(sum(temp[0:10])/10)
    #sum(temp[0:10])/10: 강남구청과 가까운 10개 주차장과의 평균거리
    #total_park = [강남구청과 가까운 10개 주차장과의 평균거리,
    #              강남역과 가까운 10개 주차장과의 평균거리,
    #                              ...                        ]
    

#%%
# 공급 접근성(SF, Supply Factor) 생성 및 sub_location.csv 업데이트

import numpy as np
from sklearn.preprocessing import StandardScaler

# 창고까지 거리 + 주차장까지 거리 합산 (짧을수록 공급 접근성이 좋음)
# 부호를 반전하여 짧은 거리 = 높은 점수로 변환 후 표준화
raw_sf = np.array(k_temp) + np.array(total_park)
# 같은 위치끼리 더한다
sf_scaled = StandardScaler().fit_transform(-raw_sf.reshape(-1, 1)).flatten()
# raw_sf = [[1],[2],[3],[4]] , reshape(행,열): -1은 알아서 계산하라는 뜻
# raw_sf.reshape(-1,1) : [[1]
#                         [2]
#                         [3]
#                         [4]]
# StandardScaler : 정규화 -> 2차원의 데이터구조여야 한다
#                         -> reshape로 2차원의 데이터구조로 만든 후 
#                         -> flatten으로 다시 1차원의 구조로 만듦
# fit() : 기준(평군,표준편차)를 계산해서 기억
# transform() : 기억한 기준으로 데이터를 변환
# 데이터 마다 기준값이 다르기 때문으로 fit으로 학습을 먼저 해주어야 한다.
import os

loc_path = BASE_DIR / "data/processed/sub_location.csv"
tmp_path = BASE_DIR / "data/processed/sub_location_tmp.csv"

data_loc = pd.read_csv(loc_path, encoding='utf-8')
data_loc['SF'] = sf_scaled
data_loc.to_csv(tmp_path, index=False, encoding='utf-8-sig')
os.replace(tmp_path, loc_path)

print("SF 업데이트 완료")
print(data_loc[['name_subway', 'SF']].head(10).to_string())

# %%
