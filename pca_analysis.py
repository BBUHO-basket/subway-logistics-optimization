import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
data = pd.read_csv(BASE_DIR / "data/raw/data_region.csv")
data = data.drop(['지역', '0~9세', '10~19세', '20~29세','30~39세', '40~49세', '50~59세', '60~69세','70~79세','80~89세','90~99세','100세 이상'],  axis=1)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

std_df = StandardScaler().fit_transform(data)
std_df = pd.DataFrame(std_df, index=data.index, columns=data.columns)
data = std_df

pca = PCA(n_components=3, random_state=42)
pca_array = pca.fit_transform(data)

# MF (Market Factor):    인구 규모 (주문인구수, 인구, 대학교 학령인구) — 48.2%
# EF (Economic Factor):  경제 수준 (지역내총생산, 1인당 GRDP, 수준지수) — 22.1%
# TF (Transport Factor): 교통 활동량 (운행거리, 운행대수, 운행 총시간) — 7.6%
pca_df = pd.DataFrame(
    pca_array,
    columns=["MF", "EF", "TF"]
)

plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.show()
