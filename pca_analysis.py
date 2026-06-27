import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
data = pd.read_csv(BASE_DIR / "data/raw/data_region.csv")
index_1 = data['지역'] 
data = data.drop(['지역', '0~9세', '10~19세', '20~29세','30~39세', '40~49세', '50~59세', '60~69세','70~79세','80~89세','90~99세','100세 이상'],  axis=1)

from sklearn.preprocessing import StandardScaler

std_df = StandardScaler().fit_transform(data)
std_df = pd.DataFrame(std_df, index=data.index, columns=data.columns)
data = std_df
data.head()

# 모든 독립변수 개수(8개)만큼 주성분 개수 설정해서 분석하기
from sklearn.decomposition import PCA

pca = PCA(n_components=3, random_state=42)
pca_fit = pca.fit(data)
pca_array = pca_fit.transform(data)
pca_df = pd.DataFrame(pca_array)
pca_df.head()

name_explain = pca.components_

pca_df = pd.DataFrame(
    pca_array,
    columns=["PC1", "PC2", "PC3"]
)

data_region_pca = pd.concat([index_1, pca_df], axis=1)

print(data_region_pca.head())

import matplotlib.pyplot as plt 

plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explaind Variance')
plt.plot(pca.explained_variance_ratio_, 'o-')

# 일단 3가지로 나눔 

# pc1의 경우 총 생산 지수, 수준 지수, 지역내 총생산량 네이밍 해야할 필요 있음 

# pc3의 경우 구별 물동량 데이처 / 

# 구별 물동량/ 1인당 지역내 총 생산량

#%% 
data_s = pd.read_csv(BASE_DIR / "data/raw/data_subway.csv")
data_s = data_s.replace('-','nan')
data_s = data_s.replace('#DIV/0!','nan')
data_s = data_s.replace('nan',0)
data_s = data_s.fillna(0) 

data_r = pd.read_csv(BASE_DIR / "data/raw/data_region.csv")

for i in range(len(data_s)):
    for j in range(len(data_r)):    
        if data_s.iloc[i, 2] == data_r.iloc[j, 0] :
            data_s.iloc[i, 10] = data_r.iloc[j, 29] 
            data_s.iloc[i, 11] = data_r.iloc[j, 30]             
            data_s.iloc[i, 12] = data_r.iloc[j, 31] 

# 주문 인구수와 연령병 통계 비율 산포도 분석 


#%% 
# 코사인 유사도 bmart와 실제 지하철역과의 관계
# bmart 
data_b = pd.read_csv(BASE_DIR / "data/raw/data_bmart.csv")
temp_index = data_b['지점 이름 ']

temp1 = data_b.iloc[:,4:6].to_numpy()
# b마트의 위치 
temp2 = data_s.iloc[:,4:6].to_numpy()
# 주변 역의 위치



from haversine import haversine

temp1 = data_b[["Latitude", "Longitude"]].astype(float).to_numpy()
temp2 = data_s[["Latitude", "Longitude"]].astype(float).to_numpy()

total = []

for i in temp2:
    temp_list = []
    for j in temp1:
        dist = haversine(i, j, unit="km")
        temp_list.append(dist)
    
    total.append(temp_list)

print("거리 계산 완료")
print("지하철역 개수:", len(temp2))
print("B마트 지점 개수:", len(temp1))
print("총 거리 계산 개수:", len(temp2) * len(temp1))
# 지하철역의 특성 및 관계 

#%% 
# 군집 분석


data_g = pd.read_csv(BASE_DIR / "data/raw/data_subway_군집.csv")
X = data_g.iloc[:,1:]

X_s = StandardScaler().fit_transform(X)
X

import matplotlib.pyplot as plt

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X.iloc[:,0],X.iloc[:,1], label='True Position')

for label, x, y in zip(labels, X.iloc[:, 0], X.iloc[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#%% 

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(X, 'ward')


plt.figure(figsize=(100, 50))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

#%% 

from sklearn.cluster import AgglomerativeClustering 
# 병합 군집으로 봤을때 

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
temp_cluter = cluster.fit_predict(X)

plt.figure(figsize=(10, 7))
plt.scatter(X['Latitude'], X['Longitude'], c=cluster.labels_, cmap='rainbow')

pca_sub = PCA(n_components=3).fit(X_s)
loadings = pd.DataFrame(
    pca_sub.components_,
    columns=X.columns,
    index=['PC1','PC2','PC3']
)

print(loadings)