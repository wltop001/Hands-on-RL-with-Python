"""
以批（batch）为单位处理数据
"""
import numpy as np
from tensorflow.python import keras as K

# 2层神经网络
model = K.Sequential([
    K.layers.Dense(units=4, input_shape=(2, ), activation='sigmoid'),
    K.layers.Dense(units=4)
])

# 使用大小为3的批数据（维度为2）
batch = np.random.rand(3, 2)

y = model.predict(batch)
print(y.shape)  # 变成（3,4）
