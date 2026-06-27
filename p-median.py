# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:21:14 2022

@author: yoonshin
"""

#%% 
# 모듈
from operator import countOf
import pandas as pd
import numpy as np 
from haversine import haversine
from pulp import *
import time
import folium
import random


#%% 
# 데이터 전처리

from pathlib import Path
BASE_DIR = Path(__file__).parent
raw_path = BASE_DIR / "data" / "raw"
proc_path = BASE_DIR / "data" / "processed"
output_path = str(BASE_DIR / "output") + "/"
demand_data = pd.read_csv(proc_path / "demand_location.csv")
sub_data = pd.read_csv(proc_path / "sub_location.csv")

per_y = sub_data['y']

demand_xy = demand_data[["Latitude","Longitude"]].to_numpy()
sub_xy = sub_data[["Latitude","Longitude"]].to_numpy()

num_de = len(demand_xy)
num_sub = len(sub_xy)

distance = np.zeros((num_de, num_sub), dtype="f")
distance2 = np.zeros((num_de, num_sub), dtype="f")
distance3 = np.zeros((num_de, num_sub), dtype="f")

alpha = 1

h = demand_data['demand'].to_numpy()

for sub in range(0, num_de) :
    for j in range(0, num_sub) :
        distance[sub][j] = haversine(demand_xy[sub], sub_xy[j])
        distance2[sub][j] = h[sub]*haversine(demand_xy[sub], sub_xy[j]) 

facilities = list(range(0,num_sub))
demand = list(range(0,num_de))
distance_d = pd.DataFrame(distance2)

with open(raw_path / "district.json", encoding='utf8') as js:
    geo = json.loads(js.read())
    
"""
distance_d에 대한 파라미터 조절 


"""

#%% 
# 기존의 p-median 식
"""
기존의 p-median 식
"""
p = 10

def p_median(p):
    t1=time.time()

    X = LpVariable.dicts('X',(facilities),0,1,LpInteger)
    Y = LpVariable.dicts('Y', (demand,facilities),0,1,LpInteger)

    prob = LpProblem('P_Median', LpMinimize)

    prob += lpSum(lpSum(distance_d.iloc[i,j] * Y[i][j] for j in facilities) for i in demand)

    for sub in demand: prob += lpSum(Y[sub][j] for j in facilities) == 1

    prob += lpSum([X[j] for j in facilities]) == p

    for sub in demand:
        for j in facilities:
            prob +=  Y[sub][j] <= X[j]

    prob.solve()

    print("Processing time took:",time.time()-t1)
    return prob

prob1 = p_median(p)

#%%
# 개선된 p-median 식

def new_p_median(p, alpha):

    t1=time.time()

    # declare facility variables
    X = LpVariable.dicts('X',(facilities),0,1,LpInteger)

    # declare demand variables
    Y = LpVariable.dicts('Y', (demand,facilities),0,1,LpInteger) 

    # create the LP object, set up as a MINIMIZATION problem
    prob = LpProblem('P_Median', LpMinimize)

    # prob += lpSum(lpSum(D[i][j] * Y[i][j] for j in location) for i in demand)
    # pandas iloc looks up values by row(i) and column(j)
    prob += lpSum(lpSum(distance_d.iloc[i,j] * Y[i][j] - per_y[j]*X[j]*alpha for j in facilities) for i in demand)

    for sub in demand: prob += lpSum(Y[sub][j] for j in facilities) == 1

    prob += lpSum([X[j] for j in facilities]) == p

    for sub in demand:
        for j in facilities: 
            prob +=  Y[sub][j] <= X[j]
            
    prob.solve()
    print("Processing time took:",time.time()-t1)
    return prob
prob2 = new_p_median(p, alpha)

#%% 
#  결과 처리
def get_connections(prob):
    print(' ')
    print("Status:",LpStatus[prob.status])
    print(' ')
    print("Objective: ",value(prob.objective))
    print(' ')

    answer_x = []

    for v in prob.variables():
        subV = v.name.split('_')
        
        if subV[0] == "X" and v.varValue == 1: 
            print('p-Median Node: ', subV[1])
            answer_x.append(int(subV[1]))

    result = []
    connections = [[] for i in facilities]
    print(' ')
    for v in prob.variables():
        subV = v.name.split('_')
        if subV[0] == "Y" and v.varValue == 1: print(subV[1], ' is connected to', subV[2])

    print(' ')
    for v in prob.variables():
        subV = v.name.split('_')
        if subV[0] == "Y" and v.varValue == 1: 
            result.append([int(subV[1]), int(subV[2])])
            connections[int(subV[2])].append(int(subV[1]))

    #print(result)
    #print(connections)

    result_connect = np.array(result)
    return connections

connections1 = get_connections(prob1)
connections2 = get_connections(prob2)

#%%
# 시각화

def visualize(name, connections):
    m = folium.Map(location=[37.493236, 127.056687], zoom_start= 12)

    folium.GeoJson(geo,lambda x: {"weight":1, "fill":False, "color":'#000000'}).add_to(m)


    for sub, con in enumerate(connections):
        count = len(con)
        if count == 0: continue
        
        s_name, s_pos = sub_data.iloc[sub]['name_subway'], list(sub_xy[sub])
        
        color = '#' + ''.join(random.choice("456789AB") for _ in range(6))
        inside_count = 0
        for dem in con:
            d_name, d_pos = demand_data.iloc[dem]['name_demand'], list(demand_xy[dem])
            dist = distance[dem,sub]
            if dist <= 3:
                inside_count+=1
                folium.Circle(d_pos,radius=50 + h[dem] * 3,popup=folium.Popup(f"거리: {dist:.2f}km<br>{d_name}<br>수요량: {h[dem]}", max_width=200),fill=True, color="#0000FF", weight=1).add_to(m)
                folium.PolyLine((d_pos,s_pos),color=color,weight=4).add_to(m)
            else:
                folium.Circle(d_pos,radius=50 + h[dem] * 3,popup=folium.Popup(f"거리: {dist:.2f}km<br>{d_name}<br>수요량: {h[dem]}", max_width=200),fill=True, color="#000000", weight=1).add_to(m)

        folium.Circle(s_pos,radius=40 * inside_count + 50,popup=folium.Popup(f'{s_name}<br>총 {count}개의 연결 중<br>{inside_count}개 수요 충족중', max_width=200),fill=True, color="#FF0000", weight=4).add_to(m)
        folium.Circle(s_pos,radius=3000,fill=False, color='#000000', weight=1).add_to(m)
        
    m.save(output_path+name)

visualize('output_서울시전체_기존.html', connections1)
visualize('output_서울시전체_신규.html', connections2)
# %%
# 다양한 alpha 처리

alphas = [0.5, 0.8, 1, 10]
for alpha in alphas:
    prob = new_p_median(p,alpha)
    connections = get_connections(prob)
    visualize(f'output_서울시전체_신규_alpha_{alpha}.html', connections)
    print(f'{alpha} alpha 처리 완료')
    