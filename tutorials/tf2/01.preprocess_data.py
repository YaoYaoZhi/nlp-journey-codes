import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载训练数据集
training_data_df = pd.read_csv("./dataset/sales_data_training.csv")

# 加载测试数据集
test_data_df = pd.read_csv("./dataset/sales_data_testing.csv")

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))

# 对训练集和测试集都做同样的归一化处理：输入和输出都做了归一化
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

# 打印scaler在total_earnings这一列的调整值: multiplying by 0.0000036968 and adding -0.115913
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

# 创建新的data frame
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# 保存归一化后的数据
scaled_training_df.to_csv("./dataset/sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("./dataset/sales_data_testing_scaled.csv", index=False)