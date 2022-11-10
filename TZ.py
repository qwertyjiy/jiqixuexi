# -*- coding:utf-8 -*-
import tsfel
import pandas as pd

# 加载数据
path_file = 'train_samples.csv'
data_ = pd.read_csv(path_file)

# 必要的话，利用时频域分析手段构造一些特征用于分类。
data_features = pd.DataFrame()
data_labels = []
for i in range(data_.shape[0]):
  signal_i = data_.iloc[i,:-1]
  labels_i = data_.iloc[i,-1]
  cfg_file = tsfel.get_featuuuures_by_domain()
  features_i = tsfel.time_series_features_extractor(cfg_file, signal_i, fs=1, window_size=512)  #非常耗时
  data_features = pd.concat([data_features, features_i])
  data_labels.append(int(labels_i))
  # if i==10:
  #   break

data_features['label'] = data_labels
print(data_features.head(10))
data_features.to_csv('Data_features.csv')