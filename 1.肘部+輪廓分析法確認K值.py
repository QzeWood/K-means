# Date： 2023/04/24 13:13
# Author: Mr. Q
# Introduction：肘部法确认k值


import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

start_time = time.time()
print(start_time)
print("======================================")
# 读取CSV文件
# raw_data = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/information.csv")
raw_data = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/Reward_type1.csv")
print(raw_data.dtypes)
#設定使用數據的範圍
# use_data = raw_data[['BSC_age', 'Game_age', 'BSC_I', 'Game_I', 'M_play_1', 'M_consume_2', 'M_repurchase_3', 'R_play', 'R_consume', 'R_repurchase',\
#            'F_play', 'F_consume', 'F_repurchase']]
use_data = raw_data[['R_game', 'F_play', 'M1_consume', 'M2_repurchase']]

# 将数据转换成numpy数组
X = np.array(use_data)

# 肘部方法和轮廓分析
sse = []
sil = []
for k in range(2, 11):
   kmeans = KMeans(n_clusters=k)
   kmeans.fit(X)
   sse.append(kmeans.inertia_)
   sil_score = silhouette_score(X, kmeans.labels_)
   sil.append(silhouette_score(X, kmeans.labels_))
   print("K =", k, "SSE =", kmeans.inertia_, "Silhouette score =", sil_score)

# 绘制图形
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(range(2, 11), sse)
axs[0].set_title('Elbow Method-1')
axs[0].set_xlabel('Number of clusters')
axs[0].set_ylabel('SSE')

axs[1].plot(range(2, 11), sil)
axs[1].set_title('Silhouette Analysis-1')
axs[1].set_xlabel('Number of clusters')
axs[1].set_ylabel('Silhouette score')
import os
os.system('say "hey baby kiss my ass"')
plt.show()
print("======================================")
# 打印最终汇总的SSE和Silhouette score
print("Final SSE:", sse)
print("Final Silhouette score:", sil)
print("ok!!")

end_time = time.time()
print(end_time)
duration = end_time - start_time
print(f"程式執行時間為 {duration:.2f} 秒")
