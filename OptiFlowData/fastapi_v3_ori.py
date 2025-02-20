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

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# J:1, D:4, L:11

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print(f'다음 기기로 학습 : {device}')

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

with open('./model/best_polynomial_regression_model.pkl', 'rb') as f:
  model_poly, poly = pickle.load(f)

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

# test_pred, test_out = predict_lstm('2024-10-01T00:00', model_lstm)
# print(test_pred)
# test_pred_xgb, test_out_xgb = predict_xgb('2024-10-01T00:00', model_xgb)
# print(test_pred_xgb)

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

def optimize_pump_flow(data, outflow, v_initial, capacity, max_flow = 250):
  start_time = data[0]['time']
  minutes = len(data) * 60
  pump_flow = np.zeros(minutes)  # 각 분당 펌프 유량 설정
  over_flow = np.zeros(minutes)  # 넘으면 저장
  lower_flow = np.zeros(minutes)  # 모자라면 저장
  storage = v_initial  # 초기 배수지 저장량
  storage_arr = []
  v_min, v_max = capacity * 0.55, capacity * 0.9 # 1% 보수적 한계
  
  hourly_flow = 0

  for minute in range(minutes):
    if minute % 60 == 0:  # 1시간 마다 유량 조정
      
      t = minute // 60 
      current_time = start_time + timedelta(hours=t)
      season, period = get_season_and_period(current_time)

      start_idx = minute
      end_idx = start_idx + 60
      # print(outflow.shape)
      expected_outflow = np.sum(outflow[:]) / 60

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

    pump_flow[minute] = hourly_flow
    
    storage += (pump_flow[minute] - outflow[minute // 60]) / 60
    storage_arr.append(storage)

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

  adjustment = [hourly_over_amount[i] + hourly_lower_amount[i] for i in range(24)]
  print(len(adjustment))

  water_level = storage_arr[::60]
  water_level = [adjustment[i] + water_level[i] for i in range(24)]
  water_percentage = [(level / capacity) * 100 for level in water_level]

  for i, value in enumerate(adjustment):
    if value == 0: continue
    else:
      pump_flow[i * 60 : (i + 1) * 60] += (value / 60)

  flow = pump_flow[::60]
  opti_arr = make_return_form(flow, start_time)
  final_result = make_return_form_with_height(flow, water_percentage, start_time)

  return opti_arr, final_result

# test_flow = optimize_pump_flow(test_pred, test_out, 1419, 2000)
# print(test_flow)

def calculate_daily_cost_by_linear(data):
  total_cost = []
  poly = PolynomialFeatures(3)
  for row in data:
    season, period = get_season_and_period(row['time'])
    flux_poly = poly.fit_transform(np.array([[row['value']]]))
    power_kW = model_poly.predict(flux_poly)[0]
    cost_per_kWh = electricity_rates[season][period]
    total_cost.append(int(power_kW * cost_per_kWh))
  return total_cost

# test_charge = calculate_daily_cost_by_linear(test_flow)
# test_charge_formatted = format(int(test_charge), ',')
# print(test_charge_formatted)

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

SPRINGBOOT_URL = os.environ.get('SPRINGBOOT_URL', 'http://10.125.121.226:8080/api/save')  # Spring Boot 엔드포인트

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
  
  capacity = {'d' : 1000, 'j' : 2000, 'l' : 1200}
  _, optiflow_lstm = optimize_pump_flow(prediction_lstm, outflow_lstm, input_data.waterLevel, capacity[input_data.name])
  print(optiflow_lstm)
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

  capacity = {'d' : 1000, 'j' : 2000, 'l' : 1200}
  _, optiflow_xgb = optimize_pump_flow(prediction_xgb, outflow_xgb, input_data.waterLevel, capacity[input_data.name])

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
  print(future_data.shape)
  start_time = future_data.iloc[0]['observation_time']
  print('start_time')
  if len(future_data) != 24:
    logger.error(f'Not enough future data found. Length: {len(future_data)}')
    raise ValueError('이후 24개의 데이터를 찾을 수 없습니다.')

  capacity = {'d' : 1000, 'j' : 2000, 'l' : 1200}

  flow_truth = make_return_form(future_data['input'], start_time)
  cost_truth = calculate_daily_cost_by_linear(flow_truth)
  flow_opti, _ = optimize_pump_flow(prediction_xgb, outflow_xgb, input_data.waterLevel, capacity[input_data.name])
  cost_opti = calculate_daily_cost_by_linear(flow_opti)

  cost_truth_form = make_return_form(cost_truth, start_time)
  cost_opti_form = make_return_form(cost_opti, start_time)
  
  result = {'truth' : cost_truth_form, 'optimization' : cost_opti_form}
  print(result)
  # result = {'truth' : [1, 2], 'optimization' : [3, 4]}
  return result