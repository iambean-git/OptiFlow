from preamble import *

import pickle

with open('best_polynomial_regression_model.pkl', 'rb') as f:
  model, poly = pickle.load(f)

file_path = './data/d_pressure_flux_outflow.csv'

data = pd.read_csv(file_path, index_col='datetime')
data.index = pd.to_datetime(data.index)
hourly_data = data.resample('h').sum()
hourly_data.reset_index(inplace=True)
hourly_data_one_day = hourly_data[:24]

hourly_data.head()

from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, time

electricity_rates = {
  'summer': {
    'off_peak': 64.37,  # 경부하 요금 (원/kWh)
    'mid_peak': 92.46,  # 중간부하 요금 (원/kWh)
    'on_peak': 123.88   # 최대부하 요금 (원/kWh)
  },
  'spring_fall': {
    'off_peak': 64.37,  # 경부하 요금 (원/kWh)
    'mid_peak': 69.50,  # 중간부하 요금 (원/kWh)
    'on_peak': 86.88    # 최대부하 요금 (원/kWh)
  },
  'winter': {
    'off_peak': 71.88,  # 경부하 요금 (원/kWh)
    'mid_peak': 90.80,  # 중간부하 요금 (원/kWh)
    'on_peak': 116.47   # 최대부하 요금 (원/kWh)
  }
}

# 시간대 구분
time_periods = {
  'summer': {
    'off_peak': [(time(23, 0), time(9, 0))],
    'mid_peak': [(time(9, 0), time(11, 0)), (time(12, 0), time(13, 0)), (time(17, 0), time(23, 0))],
    'on_peak': [(time(11, 0), time(12, 0)), (time(13, 0), time(17, 0))]
  },
  'spring_fall': {
    'off_peak': [(time(23, 0), time(9, 0))],
    'mid_peak': [(time(9, 0), time(11, 0)), (time(12, 0), time(13, 0)), (time(17, 0), time(23, 0))],
    'on_peak': [(time(11, 0), time(12, 0)), (time(13, 0), time(17, 0))]
  },
  'winter': {
    'off_peak': [(time(23, 0), time(9, 0))],
    'mid_peak': [(time(9, 0), time(10, 0)), (time(12, 0), time(17, 0)), (time(20, 0), time(22, 0))],
    'on_peak': [(time(10, 0), time(12, 0)), (time(17, 0), time(20, 0)), (time(22, 0), time(23, 0))]
  }
}

def get_season_and_period(dt):
  month = dt.month
  current_time = dt.time()
  
  if month in [7, 8]:
    season = 'summer'
  elif month in [11, 12, 1, 2]:
    season = 'winter'
  else:
    season = 'spring_fall'
  
  for period, times in time_periods[season].items():
    for start, end in times:
      if start <= current_time < end:
        return season, period
  return season, 'off_peak'
  
def calculate_daily_cost_by_linear(data):
  total_cost = 0
  poly = PolynomialFeatures(3)
  for _, row in data.iterrows():
    season, period = get_season_and_period(row['datetime'])
    flux_poly = poly.fit_transform(np.array([[row['flux']]]))
    power_kW = model.predict(flux_poly)[0]
    cost_per_kWh = electricity_rates[season][period]
    energy_kWh = power_kW / 60
    total_cost += energy_kWh * cost_per_kWh
    # print(power_kW)
  return total_cost

## 밥 먹고 수정해보자
def calculate_daily_cost_by_linear_hourly(data):
  hourly_cost = np.zeros(24)
  poly = PolynomialFeatures(3)
  for i, row in data.iterrows():
    season, period = get_season_and_period(row['datetime'])
    flux_poly = poly.fit_transform(np.array([[row['flux']]]))
    power_kW = model.predict(flux_poly)[0]
    cost_per_kWh = electricity_rates[season][period]
    energy_kWh = power_kW / 60

    idx = i // 60
    hourly_cost[idx] += energy_kWh * cost_per_kWh
  return hourly_cost

cost = calculate_daily_cost_by_linear_hourly(hourly_data_one_day)
cost_format = format(int(cost[0]), ',')
print(cost_format)

from datetime import timedelta

def optimize_pump_flow(data, v_initial, capacity, max_flow = 250):
  outflow = data['outflow'].values # 시간당 유출량을 data로 받음
  minutes = len(data['datetime']) * 60
  pump_flow = np.zeros(minutes)  # 각 분당 펌프 유량 설정
  over_flow = np.zeros(minutes)  # 넘으면 저장
  lower_flow = np.zeros(minutes)  # 모자라면 저장
  storage = v_initial  # 초기 배수지 저장량
  start_time = data['datetime'][0]
  v_min, v_max = capacity * 0.31, capacity * 0.94 # 1% 보수적 한계
  
  hourly_flow = 0

  for minute in range(minutes):
    if minute % 60 == 0:  # 정각마다 유량 조정
      
      t = minute // 60 
      current_time = start_time + timedelta(hours=t)
      season, period = get_season_and_period(current_time)
      expected_outflow = outflow[t] / 60

      # 저장량을 유지하기 위한 기본 필요 유입량
      hourly_flow = expected_outflow
      
      # 심야 시간 요금 절약을 위해 조정 (추가적인 충전 고려)
      if period == 'off_peak' and storage + hourly_flow - expected_outflow <= v_max:
        hourly_flow += max((v_max - (storage - expected_outflow)) * 0.10, 0) # 추가 충전
      elif period == 'mid_peak' and storage + hourly_flow - expected_outflow <= v_max:
        hourly_flow += max((v_max - (storage - expected_outflow)) * 0.05, 0) # 추가 충전
      if period == 'on_peak' and storage + hourly_flow - expected_outflow >= v_min:
        hourly_flow -= max(((storage - expected_outflow) - v_min) * 0.01, 0) # 절감 충전

      hourly_flow = max(hourly_flow, 0)
      hourly_flow = min(hourly_flow, max_flow)

    pump_flow[minute] = max(hourly_flow, 0)
    
    storage += pump_flow[minute] - (outflow[minute // 60] / 60)

    if (storage > v_max):
      over_flow[minute] = storage - v_max
    elif (storage < v_min):
      lower_flow[minute] = v_min - storage

  over_non_zero_indices =  np.nonzero(over_flow)
  lower_non_zero_indices = np.nonzero(lower_flow)

  hourly_over_amount = np.zeros(24)
  hourly_lower_amount = np.zeros(24)
  
  if len(over_non_zero_indices[0]) != 0:
    for idx in over_non_zero_indices[0]:
      h = idx // 60
      if hourly_over_amount[h] > (-1) * over_flow[idx]:
        hourly_over_amount[h] = (-1) * over_flow[idx]
  if len(lower_non_zero_indices[0]) != 0:
    for idx in lower_non_zero_indices[0]:
      h = idx // 60
      if hourly_lower_amount[h] < lower_flow[idx]:
        hourly_lower_amount[h] = lower_flow[idx]

  adjustment = hourly_over_amount + hourly_lower_amount

  for i, value in enumerate(adjustment):
    if value == 0: continue
    else:
      pump_flow[i * 60 : (i + 1) * 60] += (value / 60)

  return pump_flow

v_initial = 1419
capacity = 2000
optimized_flow = optimize_pump_flow(hourly_data_one_day, v_initial, capacity)

print(optimized_flow, len(optimized_flow))

data_opti = hourly_data_one_day.copy()
data_opti['flux'] = optimized_flow
hourly_opti_flux = data_opti.resample('h').sum()['flux']
cost_opti = calculate_daily_cost_by_linear(data_opti)

data_opti = data.copy()
data_opti['flux'] = optimized_flow
hourly_opti_flux = data_opti.resample('h').sum()
cost_opti = calculate_daily_cost_by_linear_hourly(hourly_opti_flux)
cost_format_opti = format(int(cost_opti[0]), ',')

saved_cost = cost - cost_opti
saved_cost_format = format(int(saved_cost), ',')
print(f'Total daily electricity cost: {cost_format} KRW')
print(f'Total daily electricity cost used optimization : {cost_format_opti} KRW')
print(f'Save money on electricity bills : {saved_cost_format} KRW')
print(f'electricity bill savings rate : {(1 - (int(cost_opti) / int(cost))) * 100:.2f} % 감소')