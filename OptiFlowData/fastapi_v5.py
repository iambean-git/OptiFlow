import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta, time
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import pulp as pl

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# J:1, D:4, L:11

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print(f'다음 기기로 학습 : {device}')

capacity_global = {'d' : 1000, 'j' : 2000, 'l' : 1200}

# load data : model predict할 때 사용
try:
  data_lstm_j = pd.read_csv('./data/j_lstm_data_v4.csv')
  data_lstm_j['datetime'] = pd.to_datetime(data_lstm_j['datetime'])  # 'datetime' 컬럼을 datetime 객체로 변환
except FileNotFoundError:
  print('Error: j_lstm_data_v4.csv 파일을 찾을 수 없습니다.')
  exit()
try:
  data_lstm_d = pd.read_csv('./data/d_lstm_data_v4.csv')
  data_lstm_d['datetime'] = pd.to_datetime(data_lstm_d['datetime'])  # 'datetime' 컬럼을 datetime 객체로 변환
except FileNotFoundError:
  print('Error: d_lstm_data_v4.csv 파일을 찾을 수 없습니다.')
  exit()
try:
  data_lstm_l = pd.read_csv('./data/l_lstm_data_v4.csv')
  data_lstm_l['datetime'] = pd.to_datetime(data_lstm_l['datetime'])  # 'datetime' 컬럼을 datetime 객체로 변환
except FileNotFoundError:
  print('Error: l_lstm_data_v4.csv 파일을 찾을 수 없습니다.')
  exit()
try:
  data_xgb_j = pd.read_csv('./data/j_before_feature_importance_with_datetime_v3.csv')
  data_xgb_j['datetime'] = pd.to_datetime(data_xgb_j['datetime'])
except FileNotFoundError:
  print('Error: j_before_feature_importance_with_datetime_v3.csv 파일을 찾을 수 없습니다.')
  exit()
try:
  data_xgb_d = pd.read_csv('./data/d_before_feature_importance_with_datetime_v3.csv')
  data_xgb_d['datetime'] = pd.to_datetime(data_xgb_d['datetime'])
except FileNotFoundError:
  print('Error: d_before_feature_importance_with_datetime_v3.csv 파일을 찾을 수 없습니다.')
  exit()
try:
  data_xgb_l = pd.read_csv('./data/l_before_feature_importance_with_datetime_v3.csv')
  data_xgb_l['datetime'] = pd.to_datetime(data_xgb_l['datetime'])
except FileNotFoundError:
  print('Error: l_before_feature_importance_with_datetime_v3.csv 파일을 찾을 수 없습니다.')
  exit()
try:
  data_djl = pd.read_csv('./data/reservoir_djl_hourly.csv')
  data_djl['observation_time'] = pd.to_datetime(data_djl['observation_time'])
except FileNotFoundError:
  print('Error: reservoir_djl_hourly.csv 파일을 찾을 수 없습니다.')
  exit()

def make_return_form(data, start_time):
  arr = []

  for i, value in enumerate(data):
    dic = {}
    dic['time'] = start_time + timedelta(hours=i)
    dic['value'] = float(value)
    arr.append(dic)
  
  return arr

def make_return_form_with_height(data, percentage, start_time):
  arr = []

  for i, value in enumerate(data):
    dic = {}
    dic['time'] = start_time + timedelta(hours=i)
    dic['value'] = float(value)
    dic['height'] = float(percentage[i])
    arr.append(dic)
  
  return arr

class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(LSTMModel, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    last_out = lstm_out[:, -1, :]
    out = self.fc(last_out)
    return out

input_size = 21
hidden_size = 50
output_size = 24

try:
  model_lstm_j = LSTMModel(input_size, hidden_size, output_size)
  model_lstm_j.load_state_dict(torch.load('./model/best_lstm_checkpoint_j.pt', map_location=device))
  model_lstm_j.to(device)
  model_lstm_j.eval()
except FileNotFoundError:
  print('Error: best_lstm_checkpoint_j.pt 파일을 찾을 수 없습니다.')
  exit()
except Exception as e:
  print(f'Error loading model: {e}')
  exit()
try:
  model_lstm_d = LSTMModel(input_size, hidden_size, output_size)
  model_lstm_d.load_state_dict(torch.load('./model/best_lstm_checkpoint_d.pt', map_location=device))
  model_lstm_d.to(device)
  model_lstm_d.eval()
except FileNotFoundError:
  print('Error: best_lstm_checkpoint_d.pt 파일을 찾을 수 없습니다.')
  exit()
except Exception as e:
  print(f'Error loading model: {e}')
  exit()
try:
  model_lstm_l = LSTMModel(input_size, hidden_size, output_size)
  model_lstm_l.load_state_dict(torch.load('./model/best_lstm_checkpoint_l.pt', map_location=device))
  model_lstm_l.to(device)
  model_lstm_l.eval()
except FileNotFoundError:
  print('Error: best_lstm_checkpoint_l.pt 파일을 찾을 수 없습니다.')
  exit()
except Exception as e:
  print(f'Error loading model: {e}')
  exit()

try:
  model_xgb_j = xgb.Booster(model_file='./model/24hour_xgboost_all_features_model_250212_j.json')
except FileNotFoundError:
  print('Error: 24hour_xgboost_all_features_model_250212.json 파일을 찾을 수 없습니다.')
  exit()
except Exception as e:
  print(f'Error loading model: {e}')
  exit()

try:
  model_xgb_d = xgb.Booster(model_file='./model/24hour_xgboost_all_features_model_250212_d.json')
except FileNotFoundError:
  print('Error: 24hour_xgboost_all_features_model_250212.json 파일을 찾을 수 없습니다.')
  exit()
except Exception as e:
  print(f'Error loading model: {e}')
  exit()

try:
  model_xgb_l = xgb.Booster(model_file='./model/24hour_xgboost_all_features_model_l.json')
except FileNotFoundError:
  print('Error: 24hour_xgboost_all_features_model_250212.json 파일을 찾을 수 없습니다.')
  exit()
except Exception as e:
  print(f'Error loading model: {e}')
  exit()


with open('./model/scaler_feature_lstm_j.pkl', 'rb') as f:
  loaded_scaler_feature_j = pickle.load(f)
with open('./model/scaler_target_lstm_j.pkl', 'rb') as f:
  loaded_scaler_target_j = pickle.load(f)
with open('./model/scaler_feature_lstm_d.pkl', 'rb') as f:
  loaded_scaler_feature_d = pickle.load(f)
with open('./model/scaler_target_lstm_d.pkl', 'rb') as f:
  loaded_scaler_target_d = pickle.load(f)
with open('./model/scaler_feature_lstm_l.pkl', 'rb') as f:
  loaded_scaler_feature_l = pickle.load(f)
with open('./model/scaler_target_lstm_l.pkl', 'rb') as f:
  loaded_scaler_target_l = pickle.load(f)

with open('./model/best_polynomial_regression_model.pkl', 'rb') as f:
  model_poly, poly = pickle.load(f)

def get_xgb_model_and_data(name):
  if (name == 'j'):
    return model_xgb_j, data_xgb_j
  elif (name == 'd'):
    return model_xgb_d, data_xgb_d
  elif (name == 'l'):
    return model_xgb_l, data_xgb_l

def get_lstm_model_data_and_scaler(name):
  if (name == 'j'):
    return model_lstm_j, data_lstm_j, loaded_scaler_feature_j, loaded_scaler_target_j
  elif (name == 'd'):
    return model_lstm_d, data_lstm_d, loaded_scaler_feature_d, loaded_scaler_target_d
  elif (name == 'l'):
    return model_lstm_l, data_lstm_l, loaded_scaler_feature_l, loaded_scaler_target_l

def predict_lstm(dt, name):  # dt는 datetime 객체
  print('lstm 실행')
  model, data_lstm, loaded_scaler_feature, loaded_scaler_target = get_lstm_model_data_and_scaler(name)
  
  model.eval()
  past_data = data_lstm[data_lstm['datetime'] < dt].tail(168)
  start_time = past_data.iloc[-1]['datetime'] + timedelta(hours=1)
  if len(past_data) != 168:
    raise ValueError('이전 168개의 데이터를 찾을 수 없습니다.')

  input_data = past_data.drop('datetime', axis=1).values.astype(np.float32)
  scaled_input_data = loaded_scaler_feature.transform(input_data)

  input_tensor = torch.tensor(scaled_input_data).unsqueeze(0).to(device) # (batch_size, sequence_length, input_size)

  with torch.no_grad():
    predicted = model(input_tensor)

  predicted = predicted.cpu().numpy()

  predicted_original = loaded_scaler_target.inverse_transform(predicted)

  pred_arr = make_return_form(predicted_original[0], start_time)
  print(pred_arr)
  return pred_arr, predicted_original[0]

def predict_xgb(dt, model, data):
  past_data = data[data['datetime'] < dt].tail(168)
  start_time = past_data.iloc[-1]['datetime'] + timedelta(hours=1)

  if len(past_data) != 168:
    raise ValueError('이전 168개의 데이터를 찾을 수 없습니다.')

  input_data = past_data.drop('datetime', axis=1).values.astype(np.float32)
  # print(input_data.shape)
  dtest = xgb.DMatrix(input_data.reshape(1, -1))
  pred = model.predict(dtest)
  # print(pred)
  result = make_return_form(pred[0], start_time)
  return result, pred[0]

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
      if start > end:  # 예: 23:00 ~ 09:00
        if start <= current_time or current_time < end:
          return season, period
      else:
        if start <= current_time < end:
          return season, period
  return season, 'off_peak'

def get_hourly_rates(start_datetime):
  """주어진 시작 시간부터 24시간 동안의 시간별 전기 요금을 반환합니다."""
  hourly_rates = []
  for hour in range(24):
    current_dt = start_datetime.replace(hour=(start_datetime.hour + hour) % 24)
    season, period = get_season_and_period(current_dt)
    hourly_rates.append(electricity_rates[season][period])
  return hourly_rates

def calculate_total_cost(inflow, start_datetime):
  total_cost = []

  for h in range(24):
    current_dt = start_datetime.replace(hour=(start_datetime.hour + h) % 24)
    season, period = get_season_and_period(current_dt)
    
    flux_poly = poly.transform(np.array([[inflow[h]]]))
    power_kW = model_poly.predict(flux_poly)[0]
    cost_per_kWh = electricity_rates[season][period]
    
    hourly_cost = power_kW * cost_per_kWh
    total_cost.append(hourly_cost)
  
  return total_cost

def simulate_water_levels(water_level, capacity, inflow, outflow):
  levels = [water_level]
  percentage = [round((water_level / capacity) * 100, 3)]

  for h in range(24):
    new_level = levels[-1] + inflow[h] - outflow[h]
    levels.append(new_level)
    percentage.append(round((new_level / capacity) * 100, 3))
    
  return percentage[:-1]

def optimize_inflow(water_level, capacity, outflow, start_datetime, min_flow=25, max_flow=380, smoothness_weight=0.1):
    """
    배수지 유입량 최적화 (유입량 최소/최대 제약, 스무딩)

    Args:
        water_level: 현재 저수량
        capacity: 최대 저수량
        outflow: 시간당 유출량
        start_datetime: 시작 날짜/시간
        min_flow: 최소 유입량 (기본값: 0)
        max_flow: 최대 유입량 (기본값: 380)
        smoothness_weight: 유입량 변화 스무딩 가중치 (기본값: 0)

    Returns:
        optimal_inflow: 최적 유입량
    """
    min_safe_level = capacity * 0.55
    max_safe_level = capacity * 0.9
    hourly_rates = get_hourly_rates(start_datetime)

    prob = pl.LpProblem("Reservoir_Inflow_Optimization", pl.LpMinimize)
    
    # 유입량 변수 정의: 최소/최대 유입량 제약 반영
    inflow = [pl.LpVariable(f"inflow_{h}", lowBound=min_flow, upBound=max_flow) for h in range(24)]

    # 목적 함수: 전기 요금 + 유입량 변화량 페널티
    prob += pl.lpSum([inflow[h] * hourly_rates[h] for h in range(24)])  # 전기 요금

    # 유입량 변화량 (L1 norm) 최소화 -> 스무딩 효과 (smoothness_weight > 0 인 경우)
    if smoothness_weight > 0:
        for h in range(23):
            prob += pl.lpSum([smoothness_weight * (inflow[h+1] - inflow[h])])
            prob += pl.lpSum([smoothness_weight * (inflow[h] - inflow[h+1])])
    
    # 제약 조건 1: 저수량 안전 범위 (강제)
    current_level = water_level
    for h in range(24):
        current_level += inflow[h] - outflow[h]
        prob += current_level >= min_safe_level
        prob += current_level <= max_safe_level

    # 제약 조건 2: 오전 9시 저수량 (소프트 제약)
    target_hour = 9
    hours_to_target = (target_hour - start_datetime.hour) % 24
    level_at_9am = water_level + pl.lpSum([inflow[h] - outflow[h] for h in range(hours_to_target)])
    deviation_9am = pl.LpVariable("deviation_9am", lowBound=0)
    prob += level_at_9am + deviation_9am >= max_safe_level  # max_safe_level 보다 크거나 같도록
    prob += level_at_9am - deviation_9am <= max_safe_level  # max_safe_level 보다 작거나 같도록
    penalty_weight_9am = 1000  # 9시 페널티
    prob += penalty_weight_9am * deviation_9am

    prob.solve(pl.PULP_CBC_CMD(msg=False))
    optimal_inflow = [inflow[h].value() for h in range(24)]

    return optimal_inflow

import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import logging
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse

# 로거 설정
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('uvicorn.error')  

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['http://10.125.121.219:3000', 'http://10.125.121.226:8080'],
  allow_credentials=True,
  allow_methods=['*'],  # 모든 HTTP 메서드 허용
  allow_headers=['*'],  # 모든 헤더 허용
)

# --- 요청 로깅 미들웨어 ---
@app.middleware('http')
async def log_requests(request: Request, call_next):
  logger.info(f'Request: {request.method} {request.url} - Headers: {request.headers}')
  try:
    response = await call_next(request)
  except Exception as e:
    logger.exception(f'Request failed: {e}')  # 예외 발생 시 스택 트레이스
    raise
  logger.info(f'Response status code: {response.status_code}')
  return response

# --- Exception handling (HTTPException 및 일반 Exception) ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
  logger.error(f'HTTPException: {exc.status_code} - {exc.detail}')
  return JSONResponse(
    status_code=exc.status_code,
    content={'message': exc.detail},
  )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
  logger.exception(f'Unhandled exception: {exc}')
  return JSONResponse(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    content={'message': 'Internal Server Error'},
  )

class InputData(BaseModel):
  datetime: str  # datetime 객체를 받도록 수정
  waterLevel: float
  name: str # 저수지명

@app.post('/api/predict/lstm')
async def predict_data_lstm(request: Request, input_data: InputData):
  print('lstm 요청')
  logger.info(f'Request: {request.method} {request.url}')
  # logger.info(f'Response status code: {input_data.status_code}')
  try:
    prediction_lstm, outflow_lstm = predict_lstm(input_data.datetime, input_data.name)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:  # 예외 처리 추가
    raise HTTPException(status_code=500, detail=str(e))
  
  start_datetime = prediction_lstm[0]['time']
  
  opti_inflow = optimize_inflow(input_data.waterLevel, capacity_global[input_data.name], outflow_lstm, start_datetime, max_flow=250)
  final_levels = simulate_water_levels(input_data.waterLevel, capacity_global[input_data.name], opti_inflow, outflow_lstm)
  optiflow_lstm = make_return_form_with_height(opti_inflow, final_levels, start_datetime)

  result = {'prediction' : prediction_lstm, 'optiflow' : optiflow_lstm}
  # print(result, type(result))
  return result

  # try:
  #   spring_response = requests.post(SPRINGBOOT_URL, json=result)
  #   spring_response.raise_for_status()
  # except requests.exceptions.RequestException as e:
  #   raise HTTPException(status_code=500, detail=f'Failed to communicate with Spring Boot: {str(e)}')

  # final_result = spring_response.json()
  # return final_result

@app.post('/api/predict/xgb')
async def predict_data_xgb(request: Request, input_data: InputData):
  logger.info(f'Request: {request.method} {request.url}')
  # logger.info(f'Response status code: {input_data.status_code}')
  model, data = get_xgb_model_and_data(input_data.name)
  try:
    prediction_xgb, outflow_xgb = predict_xgb(input_data.datetime, model, data)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:  # 예외 처리 추가
    raise HTTPException(status_code=500, detail=str(e))

  start_datetime = prediction_xgb[0]['time']

  opti_inflow = optimize_inflow(input_data.waterLevel, capacity_global[input_data.name], outflow_xgb, start_datetime, max_flow=250)
  final_levels = simulate_water_levels(input_data.waterLevel, capacity_global[input_data.name], opti_inflow, outflow_xgb)
  optiflow_xgb = make_return_form_with_height(opti_inflow, final_levels, start_datetime)

  result = {'prediction' : prediction_xgb, 'optiflow' : optiflow_xgb}
  # print(result, type(result))
  return result

@app.post('/api/cost')
async def calculate_cost(request: Request, input_data: InputData):
  logger.info(f'Request: {request.method} {request.url}')
  model, data = get_xgb_model_and_data(input_data.name)
  try:
    prediction_xgb, outflow_xgb = predict_xgb(input_data.datetime, model, data)
  except ValueError as e:
    logger.error(f'ValueError in predict_xgb: {e}')
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    logger.exception(f'Exception in predict_xgb: {e}')
    raise HTTPException(status_code=500, detail=str(e))

  id = {'d' : 4, 'j' : 1, 'l' : 11}
  future_data = data_djl[(data_djl['observation_time'] >= input_data.datetime) & (data_djl['reservoir_id'] == id[input_data.name])].head(24)

  start_datetime = future_data.iloc[0]['observation_time']

  if len(future_data) != 24:
    logger.error(f'Not enough future data found. Length: {len(future_data)}')
    raise ValueError('이후 24개의 데이터를 찾을 수 없습니다.')

  truth_inflow = future_data['input']
  cost_truth = calculate_total_cost(truth_inflow, start_datetime)
  opti_inflow = optimize_inflow(input_data.waterLevel, capacity_global[input_data.name], outflow_xgb, start_datetime, max_flow=250)
  cost_opti = calculate_total_cost(opti_inflow, start_datetime)

  cost_truth_form = make_return_form(cost_truth, start_datetime)
  cost_opti_form = make_return_form(cost_opti, start_datetime)
  
  result = {'truth' : cost_truth_form, 'optimization' : cost_opti_form}
  print(result)
  # result = {'truth' : [1, 2], 'optimization' : [3, 4]}
  return result