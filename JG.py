# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
# 加载模型

model = lgb.Booster(model_file='lightGBM_model_0501.txt')

X_test_features = pd.read_csv('X_test_features.csv')
test_pre_lgb = model.predict(X_test_features, num_iteration=model.best_iteration)
preds = np.argmax(test_pre_lgb, axis=1)

# 生成submint.csv文件
submit = pd.read_csv('submit_sample.csv')
submit['label'] = preds
submit.to_csv('submit.csv', index=False)
