from preamble import *
import holidays
import numpy as np
import pandas as pd

raw_data9 = pd.DataFrame(pd.read_csv('data/raw data/9.csv', header=None))
raw_data10 = pd.DataFrame(pd.read_csv('data/raw data/10.csv', header=None))
raw_data11 = pd.DataFrame(pd.read_csv('data/raw data/11.csv', header=None))
raw_data12 = pd.DataFrame(pd.read_csv('data/raw data/12.csv', header=None))

input_flow = raw_data9.set_index(1)[2]
output_flow = raw_data10.set_index(1)[2] + raw_data11.set_index(1)[2]
height = raw_data12.set_index(1)[2]

j_data = pd.DataFrame({'input_flow':input_flow, 'output_flow':output_flow, 'height':height})
j_data.index.name = 'datetime'
j_data.index = pd.to_datetime(j_data.index)

hourly_data = j_data.resample('h').sum()
hourly_data = hourly_data.reset_index()
hourly_input = round(hourly_data['input_flow'] / 60, 2)
hourly_output = round(hourly_data['output_flow'] / 60, 2)
hourly_height = round(hourly_data['height'] / 60, 2)

df1 = pd.DataFrame(pd.read_csv('./data/weather_2023.csv'))
df2 = pd.DataFrame(pd.read_csv('./data/weather_2024.csv'))
weather_2023 = df1.iloc[:, 2:]
weather_2024 = df2.iloc[:, 2:]
weather_2023.fillna(0, inplace=True)
weather_2024.fillna(0, inplace=True)

weather = pd.concat([weather_2023, weather_2024], axis=0)
weather = weather.set_index('일시').astype(float)
weather = weather.reset_index()
weather.rename(columns={'일시':'datetime', '기온(°C)': 'temperature', '강수량(mm)': 'precipitation', '풍속(m/s)': 'wind_speed', '습도(%)': 'humidity', '현지기압(hPa)':'atmospheric_pressure', '적설(cm)':'snow'}, inplace=True)
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather['outflow'] = hourly_output
weather['inflow'] = hourly_input
weather['height'] = hourly_height

df = weather.dropna()

df.loc[df['outflow'] <= 1, 'outflow'] = np.nan
median_value = df['outflow'].median()
df['outflow'] = df['outflow'].fillna(median_value)

target_variable = 'outflow'

df_temp = df.copy()
df_temp['outflow_diff'] = df_temp[target_variable].diff().abs()
limit = 0.001 # df_temp['outflow_diff'][2019] = 0.0014899999996487168

threshold = df_temp['outflow_diff'].mean() + 2 * df_temp['outflow_diff'].std()
anomaly_indices = df_temp[df_temp['outflow_diff'] > threshold].index.tolist()
 
df_replace = df_temp.drop('outflow_diff', axis=1)
for index in anomaly_indices:
  if index > 0 and index < len(df_replace) :
    df_replace.loc[index, target_variable] = df_replace.loc[index-1, target_variable]
  elif index == 0 and len(df_replace) > 1:
    df_replace.loc[index, target_variable] = df_replace.loc[index+1, target_variable]

df_cleaned = df_replace.copy()

def create_time_feature(df): 
  kr_holidays = holidays.KR()
  df['dayofmonth'] = df['datetime'].dt.day 
  df['dayofweek'] = df['datetime'].dt.dayofweek 
  df['quarter'] = df['datetime'].dt.quarter 
  df['month'] = df['datetime'].dt.month 
  df['year'] = df['datetime'].dt.year 
  df['dayofyear'] = df['datetime'].dt.dayofyear 
  df['week'] = df['datetime'].dt.isocalendar().week
  df['hour'] = df['datetime'].dt.hour
  df['minute'] = df['datetime'].dt.minute
  df['second'] = df['datetime'].dt.second
  df['is_weekend'] = df['datetime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
  df['is_holiday'] = (df['datetime'].dt.dayofweek >= 5 | df['datetime'].isin(kr_holidays)).astype(int)
  return df

df_featured = create_time_feature(df_cleaned)

df_featured['lag_1h'] = df_featured['outflow'].shift(1) # lag_1h(1시간전 배수량)
df_featured['lag_2h'] = df_featured['outflow'].shift(2) # lag_2h(1시간전 배수량)
df_featured['lag_3h'] = df_featured['outflow'].shift(3) # lag_3h(1시간전 배수량)
df_featured['lag_24h'] = df_featured['outflow'].shift(24) # lag_24h(전날 같은 시간 배수량)
df_featured['lag_168h'] = df_featured['outflow'].shift(168) # lag_168h(일주일 전 같은 시간 배수량)

df_featured['rolling_3h_avg'] = df_featured['outflow'].rolling(window=3, min_periods=1).mean() # rolling_3h_avg(직전 3시간 평균 소비량)
df_featured['rolling_6h_max'] = df_featured['outflow'].rolling(window=3, min_periods=1).max() # rolling_3h_max(직전 3시간 중 최대소비량)
df_featured['rolling_6h_avg'] = df_featured['outflow'].rolling(window=6, min_periods=1).mean() # rolling_6h_avg(직전 6시간 평균 소비량)
df_featured['rolling_6h_max'] = df_featured['outflow'].rolling(window=6, min_periods=1).max() # rolling_6h_max(직전 6시간 중 최대소비량)
df_featured['rolling_24h_max'] = df_featured['outflow'].rolling(window=24, min_periods=1).max() # rolling_24h_max(직전 24시간 중 최고치)

df_featured['change_rate_1h'] = df_featured['outflow'].pct_change() # change_rate_1h(직전 1시간 대비 변화율)
df_featured['rolling_7d_std'] = df_featured['outflow'].rolling(window=7*24, min_periods=1).std() # rolling_7d_std(지난 7일 변동성)
df_featured['delta_24h'] = (df_featured['outflow'] - df_featured['outflow'].shift(24)) / df_featured['outflow'].shift(24) # delta_24h(24시간 변화율)

# daily_range_lag_1d(이전 날 최대-최소 차이)
daily_max = df_featured.groupby(df_featured['datetime'].dt.date)['outflow'].max()
daily_min = df_featured.groupby(df_featured['datetime'].dt.date)['outflow'].min()
daily_range = daily_max - daily_min
daily_range = daily_range.reindex(df_featured['datetime'].dt.date)
df_featured['daily_range_lag_1d'] = daily_range.shift(1).values

df_featured.to_csv(path_or_buf='data/j_before_feature_importance_with_datetime_v3.csv', index=False)

df_featured.drop('datetime', axis=1, inplace=True)

df_featured.to_csv(path_or_buf='data/j_before_feature_importance_v3.csv', index=False)