import pandas as pd
import tensorflow.keras as keras

training_data_df = pd.read_csv("./dataset/sales_data_training_scaled.csv")

X = training_data_df.drop('销售总额', axis=1).values
Y = training_data_df[['销售总额']].values

# 定义模型
model = keras.Sequential()
model.add(keras.layers.Dense(50, input_dim=9, activation='relu', name='layer_1'))
model.add(keras.layers.Dense(100, activation='relu', name='layer_2'))
model.add(keras.layers.Dense(50, activation='relu', name='layer_3'))
model.add(keras.layers.Dense(1, activation='linear', name='output_layer'))
model.compile(loss='mean_squared_error', optimizer='adam')

# 构建一个日志打印组件
logger = keras.callbacks.TensorBoard(
    log_dir='logs',
    write_graph=True,
    histogram_freq=5
)

# 加载测试数据集
test_data_df = pd.read_csv("./dataset/sales_data_testing_scaled.csv")

X_test = test_data_df.drop('销售总额', axis=1).values
Y_test = test_data_df[['销售总额']].values

# 训练模型
model.fit(X, Y, epochs=50, shuffle=True, verbose=2, callbacks=[logger], validation_data=[X_test, Y_test])

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
