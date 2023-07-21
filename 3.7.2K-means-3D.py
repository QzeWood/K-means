# Date： 2023/04/24 13:13
# Author: Mr. Q
# Introduction：将最佳化的k值代入聚類模型中，並進行3D繪圖，最後將聚類結果添加到原資料中。

import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

start_time = time.time()
print(start_time)
print("======================================")
#忽略警告信息
import warnings
warnings.filterwarnings('ignore')

#解决中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# #查找自己電腦的字體，本機為Mac，選用了庫中的'Arial'字體
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['/System/Library/Fonts/supplemental/Arial.ttf']
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
pd.set_option('display.max_columns', None)  # 让pandas显示所有列

# 读取CSV文件
information = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/Reward_type2+MM.csv")
# information = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/Reward_type1.csv")
# information = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/information.csv")
# information = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/information++.csv")
# information = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/information+Zscore.csv")
# information = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/information+Rs.csv")
# data = pd.read_csv("/Users/wood/Desktop/UXLab/dataclean_v2/ML/information.csv", header=0, usecols=[11, 12, 13, 14])
#下面是選取我們使用資料的欄位資訊。
# df = information[["R_game'", "F_play", "M1_consume", "M2_repurchase"]]
# df = information[["R_game", "F_play", "M1_consume", "M2_repurchase"]]
df = information[["R_game'", "F_play'", "M1_consume'", "M2_repurchase'"]]
# df = information[["R_game(D)'", "F_play'", "M1_consume'", "M2_repurchase'"]]
information['label'] = 0
# 选取需要进行聚类的列
X = df
# 构建K-means聚类模型并进行训练
kmeans = KMeans(n_clusters=2, random_state=1).fit(X)
# 输出聚类结果并将标签添加到原数据集；
# 查看每个簇的均值
df['label'] = kmeans.labels_
print(df.groupby('label').mean())
#返回每个簇的大小（即每一群有多少人）
print('聚类分群信息：')
print(df.groupby('label').size())
# 输出聚类中心
centers = kmeans.cluster_centers_
print('聚类中心：')
print(centers)
# 对“R”，“F”，“M1”“M2”列的数据进行叙述性统计分析
print('叙述性统计分析：')
print(df.describe())

# # 绘制三维散点图
fig = plt.figure(figsize=(10, 10))
# # "R_game'", "F_play'", "M1_consume'", "M2_repurchase'"
# # RFM1的三维散点图，
# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# # ax1.scatter(df["F_play"], df["R_game'"], df["M1_consume"], c=kmeans.labels_)
# # ax1.scatter(centers[:, 1], centers[:, 0], centers[:, 2], marker='*', s=300, c='red')
# ax1.scatter(df["F_play"], df["R_game"], df["M1_consume"], c=kmeans.labels_, alpha=1)  # alpha参数控制散点的透明度
# ax1.set_xlabel("F_play")
# ax1.set_ylabel("R_game")
# ax1.set_zlabel("M1_consume")

# #RFM2的三维散点图
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# # ax2.scatter(df["F_play"], df["R_game'"], df["M2_repurchase"], c=kmeans.labels_)
# # ax2.scatter(centers[:, 1], centers[:, 0], centers[:, 3], marker='*', s=300, c='red')
# # 绘制聚类中心
# ax2.scatter(df["F_play"], df["R_game"], df["M2_repurchase"], c=kmeans.labels_, alpha=1)  # alpha参数控制散点的透明度
# ax2.set_xlabel("F_play")
# ax2.set_ylabel("R_game")
# ax2.set_zlabel("M2_repurchase")
# ax1.scatter(centers[:, 1], centers[:, 0], centers[:, 2], marker='*', s=100, c='red', zorder=1000000)  # 聚类中心
# ax2.scatter(centers[:, 1], centers[:, 0], centers[:, 3], marker='*', s=100, c='red', zorder=1000000)  # 聚类中心
# plt.show()
#標準化後的散點圖
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(df["F_play'"], df["R_game'"], df["M1_consume'"], c=kmeans.labels_)
# ax1.scatter(centers[:, 1], centers[:, 0], centers[:, 2], marker='*', s=300, c='red')
ax1.set_title('2-隨機型-RFM1模型三維散點圖')
ax1.set_xlabel("F_play'")
ax1.set_ylabel("R_game'")
ax1.set_zlabel("M1_consume'")
#RFM2的三维散点图
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(df["F_play'"], df["R_game'"], df["M2_repurchase'"], c=kmeans.labels_)
# ax2.scatter(centers[:, 1], centers[:, 0], centers[:, 3], marker='*', s=300, c='red')
ax2.set_title('2-隨機型-RFM2模型三維散點圖')
ax2.set_xlabel("F_play'")
ax2.set_ylabel("R_game'")
ax2.set_zlabel("M2_repurchase'")
plt.show()


# #下面的代碼是為了將分群結果添加到原始資料中，因為已經執行過了，所以被註解掉，如有需要，再自行調用。
# # 获取每个样本所属的聚类标签
# labels = kmeans.predict(X.drop(['label'], axis=1))
# # 将聚类标签加入到 "information" 数据框中
# information['cluster'] = labels
#
# # 删除'label'列
# information = information.drop('label', axis=1)
# # print(information)
# information.to_csv('/Users/wood/Desktop/UXLab/dataclean_v2/ML/RF_input/Training-2-MM+2類.csv', index=False)

print("======================================")
print("ok!!")

end_time = time.time()
print(end_time)
duration = end_time - start_time
print(f"程式執行時間為 {duration:.2f} 秒")

