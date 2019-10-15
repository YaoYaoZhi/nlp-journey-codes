import pandas as pd
import tensorflow.keras as keras

training_data_df = pd.read_csv("./dataset/sales_data_training_scaled.csv")

X = training_data_df.drop('销售总额', axis=1).values
Y = training_data_df[['销售总额']].values

# 定义模型：全连接网络
model = keras.Sequential()
model.add(keras.layers.Dense(50, input_dim=9, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X, Y, epochs=50, shuffle=True, verbose=2)

# 加载测试数据集
test_data_df = pd.read_csv("./dataset/sales_data_testing_scaled.csv")

X_test = test_data_df.drop('销售总额', axis=1).values
Y_test = test_data_df[['销售总额']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

X = pd.read_csv("./dataset/proposed_new_product.csv").values
prediction = model.predict(X)

# 待预测的数据
prediction = prediction[0][0]

# 重新恢复归一化之前的数值
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product : ${}".format(prediction))