# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:21:20 2022

@author: JHLee
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
data = pd.read_csv(BASE_DIR / "data/raw/AFC_분석.csv")

def change_num0(answers) :
    if answers[0:2] == "시장" : num = int(answers[3])
    else : num = 1/int(answers[3])
    return num

def change_num1(answers) :
    if answers[0:2] == "시장" : num = int(answers[3])
    else : num = 1/int(answers[3])
    return num

def change_num2(answers) :
    if answers[0:2] == "시장" : num = int(answers[3])
    else : num = 1/int(answers[3])
    return num

def change_num3(answers) :
    if answers[0:2] == "경제" : num = int(answers[3])
    else : num = 1/int(answers[3])
    return num

def change_num4(answers) :
    if answers[0:2] == "경제" : num = int(answers[3])
    else : num = 1/int(answers[3])
    return num

def change_num5(answers) :
    if answers[0:2] == "교통" : num = int(answers[3])
    else : num = 1/int(answers[3])
    return num


# pre_processing_num
data.iloc[:,0] = data.iloc[:,0].map(change_num0)
data.iloc[:,1] = data.iloc[:,1].map(change_num1)
data.iloc[:,2] = data.iloc[:,2].map(change_num2)
data.iloc[:,3] = data.iloc[:,3].map(change_num3)
data.iloc[:,4] = data.iloc[:,4].map(change_num4)
data.iloc[:,5] = data.iloc[:,5].map(change_num5)


#%% 

def AHP_4(a, b, c, d, e, f): # 요소가 4개인 경우 AHP분석 함수
    np01 = np.array([[1, a, b, c],[1/a, 1, d, e], [1/b, 1/d, 1, f],[1/c, 1/e, 1/f, 1]])
    col_sums = np01.sum(axis=0)
    np02 = np01 / col_sums[np.newaxis, :]
    row_avg = np02.mean(axis=1) # 최종중요도(가중치)
    np03 = np01.dot(row_avg[:, np.newaxis])/row_avg[:, np.newaxis]
    consistency = (np03.mean(axis=0)-4)/3 # 일관성 지수
    return row_avg, consistency 

a, b, c, d, e, f = data.iloc[0,:]
row_avg, consistency  = AHP_4(a, b, c, d, e, f)

list_consistency = []
list_row_avg = [] 


for i in range(len(data)) :
    a, b, c, d, e, f = data.iloc[i,:]
    row_avg, consistency  = AHP_4(a, b, c, d, e, f)
    list_row_avg.append(row_avg)
    list_consistency.append(consistency)
    
    print(consistency)

    
