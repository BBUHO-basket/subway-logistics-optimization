# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from haversine import haversine
from pulp import *
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

proc_path = Path(__file__).parent / 'data' / 'processed'
demand_data = pd.read_csv(proc_path / 'demand_location.csv')
sub_data    = pd.read_csv(proc_path / 'sub_location.csv')

demand_xy = demand_data[['Latitude', 'Longitude']].to_numpy()
sub_xy    = sub_data[['Latitude', 'Longitude']].to_numpy()
h         = demand_data['demand'].to_numpy()
per_y     = sub_data['y'].to_numpy()

num_de  = len(demand_xy)
num_sub = len(sub_xy)

distance  = np.zeros((num_de, num_sub))
distance2 = np.zeros((num_de, num_sub))
for i in range(num_de):
    for j in range(num_sub):
        d = haversine(demand_xy[i], sub_xy[j])
        distance[i][j]  = d
        distance2[i][j] = h[i] * d

facilities  = list(range(num_sub))
demand_idx  = list(range(num_de))
distance_d  = pd.DataFrame(distance2)
p = 10

def solve(alpha, is_original=False):
    X = LpVariable.dicts('X', facilities, 0, 1, LpInteger)
    Y = LpVariable.dicts('Y', (demand_idx, facilities), 0, 1, LpInteger)
    prob = LpProblem('P_Median', LpMinimize)

    if is_original:
        prob += lpSum(distance_d.iloc[i, j] * Y[i][j]
                      for i in demand_idx for j in facilities)
    else:
        prob += lpSum(distance_d.iloc[i, j] * Y[i][j] - per_y[j] * X[j] * alpha
                      for i in demand_idx for j in facilities)

    for i in demand_idx:
        prob += lpSum(Y[i][j] for j in facilities) == 1
    prob += lpSum(X[j] for j in facilities) == p
    for i in demand_idx:
        for j in facilities:
            prob += Y[i][j] <= X[j]

    prob.solve(PULP_CBC_CMD(msg=0))

    selected    = [j for j in facilities if X[j].varValue == 1]
    assignments = {i: j for i in demand_idx for j in facilities if Y[i][j].varValue == 1}

    within_3km    = sum(1 for i, j in assignments.items() if distance[i][j] <= 3)
    demand_within = sum(h[i] for i, j in assignments.items() if distance[i][j] <= 3)
    avg_w_dist    = sum(h[i] * distance[i][j] for i, j in assignments.items()) / sum(h)
    pure_obj      = sum(h[i] * distance[i][j] for i, j in assignments.items())

    return selected, within_3km, demand_within / sum(h) * 100, avg_w_dist, pure_obj

print("=" * 65)
print("알파별 비교 분석")
print("=" * 65)
header = "{:<12} {:>7} {:>10} {:>13} {:>14}".format(
    "모델", "3km내동수", "수요충족률%", "가중평균거리", "순수거리목적값")
print(header)
print("-" * 65)

sel, w, dr, wd, po = solve(0, is_original=True)
print("{:<12} {:>7} {:>10.1f} {:>13.3f} {:>14.1f}".format("기존", w, dr, wd, po))

for alpha in [0.5, 0.8, 1, 10]:
    sel, w, dr, wd, po = solve(alpha)
    label = "개선 a=" + str(alpha)
    print("{:<12} {:>7} {:>10.1f} {:>13.3f} {:>14.1f}".format(label, w, dr, wd, po))

print("=" * 65)
print("* 수요충족률: 3km 반경 내에 배정된 수요량 비율")
print("* 가중평균거리: 수요량으로 가중한 평균 배정 거리")
print("* 순수거리목적값: y 점수 제외, 거리 비용만 합산")
