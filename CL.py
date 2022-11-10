# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# 数据预处理
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# 读取数据
data = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
data = reduce_mem_usage(data)

### 查看数据
print(data.shape)

# 数据可视化
plt.figure(1)
for i in range(4):
  plt.plot(range(data.shape[0])[:512], data.iloc[:,i].values[:512])
  plt.show()

# 按标签拆分数据
SIGNALS = []
for i in range(4):
  signal = data.iloc[:,i].values
  nan_array = np.isnan(signal)
  not_nan_array = ~nan_array
  new_signal = signal[not_nan_array]
  SIGNALS.append(new_signal)
print(SIGNALS[0].shape)

lengths = [SIGNALS[i].shape[0] for i in range(4)]

# 基于train.csv文件生成的样本

X_test = pd.read_csv('test.csv')
columns_sample = list(X_test.columns)[1:] + ['label']

def sample_generater(SIGNALS, size, columns_sample):
  data_reset = []
  for i in range(4):
    signal_i = SIGNALS[i]
    m = random.choice(range(size))
    print(m)
    indexs_i = range(m, m + size*(int(len(signal_i)/size)-2), int(size/10))  # 此处的5可以控制样本量
    for j in indexs_i:
      sample_ = list(signal_i[j:j+size]) + [i]
      data_reset.append(sample_)
  data_reset = pd.DataFrame(data_reset, columns=columns_sample)
  print(data_reset.shape)
  return data_reset

data_reset = sample_generater(SIGNALS, size=512, columns_sample=columns_sample)


# 打乱顺序并保存
from sklearn.utils import shuffle
data_reset = shuffle(data_reset)
data_reset.to_csv('train_samples.csv', index=False)
print(data_reset.head())

temp = pd.read_csv('train_samples.csv')
print(temp.isnull().any())  # 判断有没有空值