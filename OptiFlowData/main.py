import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta, time

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print(f'다음 기기로 학습 : {device}')

try:
  data = pd.read_csv('./data/j_weather_data_v3.csv')
  data['datetime'] = pd.to_datetime(data['datetime'])  # 'datetime' 컬럼을 datetime 객체로 변환
except FileNotFoundError:
  print("Error: j_weather_data_v3.csv 파일을 찾을 수 없습니다.")
  exit()
  
try:
  data_height = pd.read_csv('./data/temp/j.csv')
  data_height['observation_time'] = pd.to_datetime(data_height['observation_time'])
except FileNotFoundError:
  print("Error: j.csv 파일을 찾을 수 없습니다.")
  exit()

def make_return_form(data, start_time):
  arr = []

  for i, value in enumerate(flow):
    dic = {}
    dic['time'] = start_time + timedelta(hours=i)
    dic['value'] = float(value)
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

input_size = 19
hidden_size = 50
output_size = 24

try:
  model = LSTMModel(input_size, hidden_size, output_size)
  model.load_state_dict(torch.load('./model/best_lstm_checkpoint_250206.pt', map_location=device))
  model.to(device)
  model.eval()
except FileNotFoundError:
  print("Error: best_lstm_checkpoint_250206.pt 파일을 찾을 수 없습니다.")
  exit()
except Exception as e:
  print(f"Error loading model: {e}")
  exit()

with open('./model/scaler_feature_lstm_250206.pkl', 'rb') as f:
  loaded_scaler_feature = pickle.load(f)
with open('./model/scaler_target_lstm_250206.pkl', 'rb') as f:
  loaded_scaler_target = pickle.load(f)

def predict(dt, model):  # dt는 datetime 객체
  model.eval()
  
  # dt를 기준으로 이전 168개의 데이터 가져오기
  past_data = data[data['datetime'] < dt].tail(168)
  start_time = past_data.iloc[-1]['datetime'] + timedelta(hours=1)
  print(dt, start_time)
  if len(past_data) != 168:
    raise ValueError("이전 168개의 데이터를 찾을 수 없습니다.")

  # 과거 데이터를 numpy array로 변환하고, float32 타입으로 변경
  input_data = past_data.drop('datetime', axis=1).values.astype(np.float32)
  scaled_input_data = loaded_scaler_feature.transform(input_data)

  # 데이터를 PyTorch Tensor로 변환하고, 모델에 입력할 수 있도록 reshape
  input_tensor = torch.tensor(scaled_input_data).unsqueeze(0).to(device) # (batch_size, sequence_length, input_size)

  with torch.no_grad():
    predicted = model(input_tensor)

  predicted = predicted.cpu().numpy()

  predicted_original = loaded_scaler_target.inverse_transform(predicted)

  pred_arr = make_return_form(predicted_original[0], start_time)

  return pred_arr, predicted_original[0]

# test = predict('2024-10-01T00:00', model)
# print(test)

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

def optimize_pump_flow(data, outflow, v_initial, capacity, max_flow = 250):
  start_time = data[0]['time']
  minutes = len(data) * 60
  pump_flow = np.zeros(minutes)  # 각 분당 펌프 유량 설정
  over_flow = np.zeros(minutes)  # 넘으면 저장
  lower_flow = np.zeros(minutes)  # 모자라면 저장
  storage = v_initial  # 초기 배수지 저장량
  v_min, v_max = capacity * 0.31, capacity * 0.94 # 1% 보수적 한계
  
  hourly_flow = 0

  for minute in range(minutes):
    if minute % 60 == 0:  # 1시간 마다 유량 조정
      
      t = minute // 60 
      current_time = start_time + timedelta(hours=t)
      season, period = get_season_and_period(current_time)

      start_idx = minute
      end_idx = start_idx + 60
      expected_outflow = np.sum(outflow[start_idx:end_idx]) / 60

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

  flow = pump_flow[::60]
  opti_arr = make_return_form(flow, start_time)

  return opti_arr

from fastapi import FastAPI, Request, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from fastapi import FastAPI, Request

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://10.125.121.219:3000", "http://10.125.121.226:8080"],
  allow_credentials=True,
  allow_methods=["*"],  # 모든 HTTP 메서드 허용
  allow_headers=["*"],  # 모든 헤더 허용
)

SPRINGBOOT_URL = os.environ.get("SPRINGBOOT_URL", "http://10.125.121.226:8080/api/save")  # Spring Boot 엔드포인트

class InputData(BaseModel):
  datetime: str  # datetime 객체를 받도록 수정
  waterlevel: float
  name: str # 저수지명

@app.post("/api/predict")
async def predict_data(request: Request, input_data: InputData):
  logger.info(f"Request: {request.method} {request.url}")
  # logger.info(f"Response status code: {input_data.status_code}")
  try:
    prediction, outflow = predict(input_data.datetime, model)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:  # 예외 처리 추가
    raise HTTPException(status_code=500, detail=str(e))

  optiflow = optimize_pump_flow(prediction, outflow, input_data.waterlevel, 2000)
  result = {"prediction" : prediction, 'optiflow' : optiflow}
  # print(result, type(result))
  return result
  # try:
  #   spring_response = requests.post(SPRINGBOOT_URL, json=result)
  #   spring_response.raise_for_status()
  # except requests.exceptions.RequestException as e:
  #   raise HTTPException(status_code=500, detail=f"Failed to communicate with Spring Boot: {str(e)}")

  # final_result = spring_response.json()
  # return final_result