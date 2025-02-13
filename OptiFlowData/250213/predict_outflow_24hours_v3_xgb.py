from preamble import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb

df = pd.read_csv('./data/j_before_feature_importance_v3.csv')

def create_sequences(data, seq_length, target_length):
  Xs = []
  ys = []
  for i in range(len(data) - seq_length - target_length + 1):
    X = data[i:(i + seq_length)].values
    y = data['outflow'].values[i + seq_length: i+ seq_length + target_length]
    Xs.append(X)
    ys.append(y)
  return np.array(Xs), np.array(ys)

SEQ_LENGTH = 168  # 과거 7일 (시간 단위)
TARGET_LENGTH = 24 # 이후 24시간 예측

X, y = create_sequences(df, SEQ_LENGTH, TARGET_LENGTH)

train_size = int(len(X) * 0.7)
valid_size = int(train_size * 0.1)

X_train, X_test = X[:(train_size - valid_size)], X[train_size:]
X_valid, y_valid = X[(train_size - valid_size):train_size], y[(train_size - valid_size):train_size]
y_train, y_test = y[:(train_size - valid_size)], y[train_size:]

def get_feature_importance(X_train, y_train, X_valid, y_valid, feature_names):
  reg = xgb.XGBRegressor(n_estimators=1000, eval_metric="rmse", random_state=1021, early_stopping_rounds=10)
  reg.fit(
    X_train.reshape(X_train.shape[0], -1), 
    y_train, # reshape 하지 않고 y_train 그대로 사용
    eval_set=[(X_valid.reshape(X_valid.shape[0], -1), y_valid)], # reshape 하지 않고 y_valid 그대로 사용
    verbose=10,
  )
  feature_importances = reg.feature_importances_
  
  # 순서에 맞게 feature name을 확장
  feature_names_extended = []
  for i in range(SEQ_LENGTH):
    for name in feature_names:
      feature_names_extended.append(f"{name}_t-{SEQ_LENGTH-i-1}")
          
  importance_df = pd.DataFrame({'Feature': feature_names_extended, 'Importance': feature_importances})
  importance_df = importance_df.sort_values('Importance', ascending=False)
  return reg, importance_df

feature_names = df.columns
reg_all, importance_df = get_feature_importance(X_train, y_train, X_valid, y_valid, feature_names)

y_pred_all = reg_all.predict(X_test.reshape(X_test.shape[0], -1))
rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))

print(f"Test RMSE with all features: {rmse_all}")

from sklearn.metrics import mean_squared_error, r2_score
def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse = np.sqrt(mean_squared_error(y_test, y_pred_all))
r2 = r2_score(y_test, y_pred_all)
mape = mean_absolute_percentage_error(y_test, y_pred_all)
print(f'Test RMSE: {rmse:.4f}')
print(f'Test MAPE: {mape:.4f}%')
print(f'Test R-squared: {r2:.4f}')

model_filename = './model/24hour_xgboost_all_features_model_250212_j.json'
reg_all.save_model(model_filename)

model = xgb.Booster(model_file=model_filename2)

# 시각화를 위한 날짜 데이터
data = pd.read_csv('./data/j_before_feature_importance_with_datetime_v3.csv')

start_index_test = train_size + SEQ_LENGTH
end_index_test = start_index_test + TARGET_LENGTH * 7
datetime_range = data['datetime'][start_index_test:end_index_test].values

dtest = xgb.DMatrix(X_test.reshape(X_test.shape[0], -1))
y_pred = model.predict(dtest)

# 시각화
plt.figure(figsize=(20, 8))

plt.plot(datetime_range, y_test[0:168:24].reshape(-1), label='Actual')
plt.plot(datetime_range, y_pred[0:168:24].reshape(-1), label='Predicted')

plt.xticks([])  # x축 눈금 제거
plt.xlabel('Datetime')
plt.ylabel('Outflow')
plt.title('Actual vs Predicted Outflow (24 Hour Forecast for one Day)')
plt.legend(loc='upper right') # legend 위치 조정
plt.tight_layout()        # 레이아웃 조정
plt.show()