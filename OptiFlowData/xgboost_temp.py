import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드 및 전처리
df = pd.read_csv('./data/j_weather_data_v2.csv')
df = df.drop('snow', axis=1)
df['datetime'] = pd.to_datetime(df['datetime'])
print(type(df), df.shape)

df.loc[df['outflow'] <= 1, 'outflow'] = np.nan
median_value = df['outflow'].median()
df['outflow'] = df['outflow'].fillna(median_value)

def create_time_feature(df): 
  df['dayofmonth'] = df['datetime'].dt.day 
  df['dayofweek'] = df['datetime'].dt.dayofweek 
  df['dayofyear'] = df['datetime'].dt.dayofyear 
  df['hour'] = df['datetime'].dt.hour
  df['minute'] = df['datetime'].dt.minute
  df['second'] = df['datetime'].dt.second
  return df

df = create_time_feature(df)
df = df.drop('datetime', axis=1)

# Sequential Dataset 생성 함수
def create_sequences(data, seq_length, target_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - target_length + 1):
        x = data[i:(i + seq_length)].drop('outflow', axis=1).values
        y = data['outflow'].values[i + seq_length: i+ seq_length + target_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 파라미터 설정
SEQ_LENGTH = 168  # 과거 7일 (시간 단위)
TARGET_LENGTH = 24 # 이후 24시간 예측

# Sequential Dataset 생성
X, y = create_sequences(df, SEQ_LENGTH, TARGET_LENGTH)

# Train/Test 분할
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# XGBoost 모델 학습
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train.reshape(X_train.shape[0], -1), y_train.reshape(y_train.shape[0], -1), verbose = False)

# feature importance 시각화
xgb.plot_importance(reg)
plt.show()

# 예측 함수
def predict_future(model, last_window, target_length):
    # 마지막 윈도우를 입력으로 사용하여 예측
    last_window_reshaped = last_window.reshape(1, -1)
    predictions = model.predict(last_window_reshaped)

    # 예측 결과를 원하는 길이로 자르기
    return predictions.reshape(target_length)

# Test 데이터 예측
y_pred = np.array([predict_future(reg, X_test[i], TARGET_LENGTH) for i in range(len(X_test))])

# 예측 결과 평가
mse = mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1))
rmse = np.sqrt(mse)
r2 = r2_score(y_test.reshape(-1), y_pred.reshape(-1))
print(f'MSE : {mse}, RMSE_ORI : {rmse} R-value : {r2}')