# -*- coding:utf-8 -*-
import tsfel
import pandas as pd
# 对测试集进行预测
X_test = pd.read_csv('test.csv')
# 特征工程

X_test_features = pd.DataFrame()
for i in range(X_test.shape[0]):
  signal_i = X_test.iloc[i,1:]
  cfg_file = tsfel.get_features_by_domain()
  features_i = tsfel.time_series_features_extractor(cfg_file, signal_i, fs=1, window_size=512)  #非常耗时
  X_test_features = pd.concat([X_test_features, features_i])
  # if i==10:
  #   break

print(X_test_features)
X_test_features.to_csv('X_test_features.csv')
