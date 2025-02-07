import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print(f'다음 기기로 학습 : {device}')

try:
  data = pd.read_csv('./data/j_weather_data_v3.csv')
  data['datetime'] = pd.to_datetime(data['datetime'])
except FileNotFoundError:
  print("Error: j_weather_data_v3.csv 파일을 찾을 수 없습니다.")
  exit()

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
  model.load_state_dict(torch.load('best_lstm_checkpoint_250206.pt', map_location=device))
  model.to(device)
  model.eval()
except FileNotFoundError:
  print("Error: best_lstm_checkpoint_250206.pt 파일을 찾을 수 없습니다.")
  exit()
except Exception as e:
  print(f"Error loading model: {e}")
  exit()

with open('scaler_feature_lstm_250206.pkl', 'rb') as f:
  loaded_scaler_feature = pickle.load(f)
with open('scaler_target_lstm_250206.pkl', 'rb') as f:
  loaded_scaler_target = pickle.load(f)

def predict(dt, model):
  model.eval()
  
  past_data = data[data['datetime'] < dt].tail(168)
  start_time = past_data.iloc[-1]['datetime'] + timedelta(hours=1)

  if len(past_data) != 168:
    raise ValueError("이전 168개의 데이터를 찾을 수 없습니다.")

  input_data = past_data.drop('datetime', axis=1).values.astype(np.float32)
  scaled_input_data = loaded_scaler_feature.transform(input_data)

  input_tensor = torch.tensor(scaled_input_data).unsqueeze(0).to(device) # (batch_size, sequence_length, input_size)

  with torch.no_grad():
    predicted = model(input_tensor)

  predicted = predicted.cpu().numpy()
  predicted_original = loaded_scaler_target.inverse_transform(predicted)

  pred_arr = []

  for i, value in enumerate(predicted_original[0]):
    dic = {}
    dic["time"] = start_time + timedelta(hours=i)
    dic["value"] = float(value)
    pred_arr.append(dic)

  return pred_arr

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

# SPRINGBOOT_URL = os.environ.get("SPRINGBOOT_URL", "http://10.125.121.226:8080/api/save")  # Spring Boot 엔드포인트

class InputData(BaseModel):
  datetime: str  # datetime 객체를 받도록 수정
  # reservoir: str # 저수지명 

@app.post("/api/predict")
async def predict_data(request: Request, input_data: InputData):
  logger.info(f"Request: {request.method} {request.url}")

  try:
    prediction = predict(input_data.datetime, model)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:  # 예외 처리 추가
    raise HTTPException(status_code=500, detail=str(e))

  result = {"prediction" : prediction}
  return result
