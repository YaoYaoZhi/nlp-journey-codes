import pandas as pd
import tensorflow.keras as keras

training_data_df = pd.read_csv("./dataset/sales_data_training_scaled.csv")

# 加载训练好的模型
model = keras.models.load_model('./models/trained_model.h5')

X = pd.read_csv("./dataset/proposed_new_product.csv").values
prediction = model.predict(X)

# 待预测的数据
prediction = prediction[0][0]

# 重新恢复归一化之前的数值
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product : ${}".format(prediction))

