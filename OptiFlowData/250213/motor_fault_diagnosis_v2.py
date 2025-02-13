from preamble import *

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1-1. 데이터 로딩 및 전처리
def load_and_preprocess(data_path):
  df = pd.read_csv(data_path)

  def parse_data_array(data_str):
    data_str = re.sub(r'\s+', ' ', data_str).strip()
    if data_str == '[]':
      return None
    try:
      return np.fromstring(data_str[1:-1], sep=' ')
    except ValueError:
      print(f"Warning: Could not parse data array: {data_str}")
      return None

  df['data_array'] = df['data_array'].apply(parse_data_array)
  return df

# 1-2 특징 추출 함수
def extract_features(data_array):
  x = np.array(data_array)
  x = x[:len(x)//2]

  features = []
  features.append(np.max(x) - np.min(x))
  features.append(np.mean(x))
  if len(x) > 1:
    features.append(np.std(x, ddof=1))
  else:
    features.append(0)
  features.append(np.sqrt(np.mean(x**2)))
  if len(x) > 1 and features[3] != 0:
    features.append(np.max(np.abs(x)) / features[3])
  else:
    features.append(0)

  if len(x) > 1 and features[2] != 0:
    features.append(np.mean(((x - features[1]) / features[2])**3))
    features.append(np.mean(((x - features[1]) / features[2])**4))
  else:
    features.append(0)
    features.append(0)
  return features

# 1-3 데이터프레임에 특징 추가
def create_feature_df(df):
  feature_list = []
  for _, row in df.iterrows():
    if row['data_array'] is None:
      feature_list.append(None)
    else:
      features = extract_features(row['data_array'])
      feature_list.append(features)

  valid_features = [f for f in feature_list if f is not None]
  if valid_features:
    feature_df = pd.DataFrame(valid_features, columns=[f'feature_{i}' for i in range(7)])
  else:
    feature_df = pd.DataFrame()

  feature_df = feature_df.reset_index(drop=True)

  # 채널별 특징 결합 부분
  channel_features = []
  for channel in df['channel_id'].unique():
    channel_rows = df['channel_id'] == channel  # 조건에 맞는 행 필터링
    
    # 해당 채널의 데이터가 있고, data_array가 None이 아닌 경우만
    if channel_rows.any() and df.loc[channel_rows, 'data_array'].notna().any():
      channel_df = feature_df.loc[channel_rows & df['data_array'].notna()].copy()
      channel_df.columns = [f"{col}_{channel}" for col in channel_df.columns]
      channel_features.append(channel_df)
    else:
      channel_features.append(pd.DataFrame()) # 빈 DataFrame 추가

  if channel_features:
    feature_df = pd.concat(channel_features, axis=1)
  else:
    feature_df = pd.DataFrame()
  
  # final_df 생성
  final_df = pd.DataFrame()
  if not feature_df.empty:
    final_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
  else:
    final_df = df.copy()  # feature_df가 비어있으면 원본 df 사용
    print("Warning: No features extracted. Using original dataframe.")
  
  return final_df

# 2. 정상 범위 설정 및 레이블링
def set_normal_range_and_label(final_df, date_threshold='7D', threshold_std_warn=2, threshold_std_danger=4):
  """
  각 모터별 초기 데이터를 기반으로 정상 범위를 설정하고, 이후 데이터를 레이블링합니다.

  Args:
      final_df: 특징 데이터프레임.
      date_threshold: 정상 데이터로 간주할 기간 (예: '7D' - 7일).
      threshold_std_warn: '주의' 레벨의 표준편차 임계값.
      threshold_std_danger: '위험' 레벨의 표준편차 임계값.

  Returns:
      'failure_level' 열이 추가된 데이터프레임. (0: 정상, 1: 주의, 2: 위험, 3: 고장)
  """

  final_df['failure_level'] = 0  # 초기값은 정상(0)
  final_df['acq_date'] = pd.to_datetime(final_df['acq_date'])  # 날짜 타입 변환

  for motor_id in final_df['motor_id'].unique():
    motor_data = final_df[final_df['motor_id'] == motor_id].sort_values(by='acq_date')
    
    if motor_data.empty:
      print(f"Skipping motor {motor_id} due to empty data set.")
      continue

    # 정상 데이터 기간 설정 (처음부터 date_threshold까지)
    cutoff_date = motor_data['acq_date'].min() + pd.to_timedelta(date_threshold)
    normal_data = motor_data[motor_data['acq_date'] <= cutoff_date]
    
    # 채널별 정상 범위 계산 (data_array가 None이 아닌 행만 사용)
    for channel in motor_data['channel_id'].unique():
      channel_normal_data = normal_data[(normal_data['channel_id'] == channel) & (normal_data['data_array'].notna())]
      channel_features = channel_normal_data.filter(like=f'feature_') # 'feature_'로 시작하는 열만

      if channel_features.empty:
        print(f"Skipping channel {channel} in motor {motor_id} due to empty normal data.")
        continue

      # 결측치 처리 (SimpleImputer)
      imputer = SimpleImputer(strategy='mean')
      channel_features_imputed = imputer.fit_transform(channel_features)

      normal_mean = np.mean(channel_features_imputed, axis=0)
      normal_std = np.std(channel_features_imputed, axis=0)

      # 이후 데이터 레이블링
      channel_data = motor_data[(motor_data['channel_id'] == channel) & (motor_data['acq_date'] > cutoff_date)]

      for index, row in channel_data.iterrows():
        # data_array가 None이면 레이블링 건너뛰기
        if row['data_array'] is None:
          continue

        current_features = row.filter(like=f'feature_').values.astype(float)
        if np.any(np.isnan(current_features)): # 결측치 확인
          current_features = imputer.transform([current_features])[0]

        # 벗어난 정도 계산
        z_scores = (current_features - normal_mean) / normal_std
        max_deviation = np.max(np.abs(z_scores))  # 가장 크게 벗어난 정도


        if max_deviation > threshold_std_danger * normal_std.mean():  # 고장
          final_df.loc[index, 'failure_level'] = 3
        elif max_deviation > threshold_std_warn * normal_std.mean():  # 위험
          final_df.loc[index, 'failure_level'] = 2
        elif max_deviation > threshold_std_warn:   # 주의
          final_df.loc[index, 'failure_level'] = 1

  return final_df

data_path = './data/pms_data_decompressed.csv'
df = load_and_preprocess(data_path)
final_df = create_feature_df(df)

if final_df.empty:
  print("No data available after feature extraction.")
else:
  final_df = set_normal_range_and_label(final_df)

# 3-1. 데이터 분할
final_df = final_df.sort_values(by='acq_date')
X = final_df.drop(['motor_id', 'equipment_id', 'center_id', 'channel_id', 'acq_date', 'data_array', 'failure_level'], axis=1)
y = final_df['failure_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3-2. XGBoost 모델 학습 및 튜닝
param_grid = {
  'n_estimators': [100, 200, 300],
  'max_depth': [3, 4, 5],
  'learning_rate': [0.01, 0.1, 0.2],
  'subsample': [0.8, 1.0],
  'colsample_bytree': [0.8, 1.0],
  'gamma': [0, 0.1, 0.2],
  'reg_alpha': [0, 0.01, 0.1],
  'reg_lambda': [0, 0.01, 0.1]
}

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

model_filename = './model/motor_xgboost_model_v2.json'
best_model.save_model(model_filename)

# 3-3. 모델 평가
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 3-4. 변수 중요도
feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)