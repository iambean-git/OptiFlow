import logging
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import xgboost as xgb
from functools import lru_cache

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# J:1, D:4, L:11

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print(f'다음 기기로 학습 : {device}')

with open('./model/best_polynomial_regression_model.pkl', 'rb') as f:
  model_poly, poly = pickle.load(f)

def make_return_form(data, start_time):
  arr = []

  for i, value in enumerate(data):
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

input_size = 21
hidden_size = 50
output_size = 24

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
      if start <= current_time < end:
        return season, period
  return season, 'off_peak'

def calculate_daily_cost_by_linear(data):
  total_cost = []
  poly = PolynomialFeatures(3)
  for row in data:
    season, period = get_season_and_period(row['time'])
    flux_poly = poly.fit_transform(np.array([[row['value']]]))
    power_kW = model_poly.predict(flux_poly)[0]
    cost_per_kWh = electricity_rates[season][period]
    total_cost.append(power_kW * cost_per_kWh)
  return total_cost


#---------------------------------------------------------------------------------------------------------#  

app = FastAPI()

# CORS 설정
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://10.125.121.219:3000", "http://10.125.121.226:8080"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# 로거 설정 (Uvicorn 로깅 사용)
logger = logging.getLogger("uvicorn.error")

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
  logger.info(f"Request: {request.method} {request.url} - Headers: {request.headers}")
  try:
    response = await call_next(request)
  except Exception as e:
    logger.exception(f"Request failed: {e}")
    raise
  logger.info(f"Response status code: {response.status_code}")
  return response

# Exception handling (HTTPException 및 일반 Exception)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
  logger.error(f"HTTPException: {exc.status_code} - {exc.detail}")
  return JSONResponse(
    status_code=exc.status_code,
    content={"message": exc.detail},
  )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
  logger.exception(f"Unhandled exception: {exc}")
  return JSONResponse(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    content={"message": "Internal Server Error"},
  )

class InputData(BaseModel):
  datetime: str = Field(..., description="The observation time.")
  waterLevel: float = Field(..., ge=0, description="Water level, must be non-negative.")
  name: str = Field(..., description="Reservoir name (j, d, or l).", pattern="^[jdl]$")
  modelName: str = Field(..., description="Model name.")

  class Config:
    json_schema_extra = {
      "example": {
        "datetime": "2024-10-01T00:00:00",
        "waterLevel": 1500.0,
        "name": "j",
        "modelName": "lstm"
      }
    }

@lru_cache(maxsize=None)
def load_lstm_model(name: str):
  if name == 'j':
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('./model/best_lstm_checkpoint_j.pt', map_location=device))
    model.to(device)
    model.eval()
    return model
  elif name == 'd':
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('./model/best_lstm_checkpoint_d.pt', map_location=device))
    model.to(device)
    model.eval()
    return model
  elif name == 'l':
    model = LSTMModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('./model/best_lstm_checkpoint_l.pt', map_location=device))
    model.to(device)
    model.eval()
    return model
  else:
      raise ValueError("Invalid model name")


@lru_cache(maxsize=None)
def load_xgb_model(name: str):
  if name == 'j':
    return xgb.Booster(model_file='./model/24hour_xgboost_all_features_model_250212_j.json')
  elif name == 'd':
    return xgb.Booster(model_file='./model/24hour_xgboost_all_features_model_250212_d.json')
  elif name == 'l':
    return xgb.Booster(model_file='./model/24hour_xgboost_all_features_model_l.json')
  else:
    raise ValueError("Invalid model name")

@lru_cache(maxsize=None)
def load_scalers(name: str):
  if name == 'j':
    with open('./model/scaler_feature_lstm_j.pkl', 'rb') as f:
      loaded_scaler_feature = pickle.load(f)
    with open('./model/scaler_target_lstm_j.pkl', 'rb') as f:
      loaded_scaler_target = pickle.load(f)
    return loaded_scaler_feature, loaded_scaler_target
  elif name == 'd':
    with open('./model/scaler_feature_lstm_d.pkl', 'rb') as f:
      loaded_scaler_feature = pickle.load(f)
    with open('./model/scaler_target_lstm_d.pkl', 'rb') as f:
      loaded_scaler_target = pickle.load(f)
    return loaded_scaler_feature, loaded_scaler_target
  elif name == 'l':
    with open('./model/scaler_feature_lstm_l.pkl', 'rb') as f:
      loaded_scaler_feature = pickle.load(f)
    with open('./model/scaler_target_lstm_l.pkl', 'rb') as f:
      loaded_scaler_target = pickle.load(f)
    return loaded_scaler_feature, loaded_scaler_target

def get_lstm_model(name: str = "j"):
  return load_lstm_model(name)

def get_xgb_model(name: str = "j"):
  return load_xgb_model(name)

def get_scalers(name: str = "j"):
  return load_scalers(name)


def get_data(name: str):
  if name == 'j':
    return data_lstm_j
  elif name == 'd':
    return data_lstm_d
  elif name == 'l':
    return data_lstm_l
  else:
    raise ValueError("Invalid data name")

def get_xgb_data(name: str):
  if name == 'j':
    return data_xgb_j
  elif name == 'd':
    return data_xgb_d
  elif name == 'l':
    return data_xgb_l
  else:
    raise ValueError("Invalid data name")


def optimize_pump_flow(data, outflow, v_initial, capacity, max_flow=250):
  start_time = data[0]['time']
  hours = len(data)
  pump_flow = np.zeros(hours)  # 시간 단위로 유량 계산
  storage = v_initial
  v_min, v_max = capacity * 0.31, capacity * 0.94

  for hour in range(hours):
    current_time = start_time + timedelta(hours=hour)
    season, period = get_season_and_period(current_time)
    expected_outflow = outflow[hour]

    # 기본 유입량 설정 (저장량 유지를 목표)
    hourly_flow = expected_outflow

    # 요금제에 따른 유량 조정
    if period == 'off_peak':
      # 최대 저장량까지 채울 수 있는 추가 유량 계산
      additional_flow = max(0, (v_max - (storage + hourly_flow - expected_outflow)))
      hourly_flow += min(additional_flow, max_flow - hourly_flow) * 0.10  # 추가 유량의 일부만 반영

    elif period == 'mid_peak':
      additional_flow = max(0, (v_max - (storage + hourly_flow - expected_outflow)))
      hourly_flow += min(additional_flow, max_flow - hourly_flow) * 0.05

    elif period == 'on_peak':
      # 최소 저장량 이상 유지하도록 유량 감소
      reduction_flow = max(0, ((storage + hourly_flow - expected_outflow) - v_min))
      hourly_flow -= min(reduction_flow, hourly_flow) * 0.01

    # 유량 범위 제한
    hourly_flow = np.clip(hourly_flow, 0, max_flow)

    pump_flow[hour] = hourly_flow
    storage += hourly_flow - expected_outflow

    # 저장량 한계 초과/미달 처리
    if storage > v_max:
      pump_flow[hour] -= (storage - v_max) # 넘친 만큼 유량 감소
      storage = v_max
    elif storage < v_min:
      pump_flow[hour] += (v_min - storage) # 부족한 만큼 유량 증가
      storage = v_min

    pump_flow[hour] = np.clip(pump_flow[hour], 0, max_flow) # 다시 유량 범위 제한.

  opti_arr = make_return_form(pump_flow, start_time)
  return opti_arr

@app.post("/api/predict/lstm")
async def predict_data_lstm(input_data: InputData, model: LSTMModel = Depends(get_lstm_model), scaler_feature: object = Depends(get_scalers), scaler_target: object = Depends(get_scalers), data: pd.DataFrame = Depends(get_data)):

  try:
    model.eval()
    past_data = data[data['datetime'] < input_data.datetime].tail(168)
    start_time = past_data.iloc[-1]['datetime'] + timedelta(hours=1)

    if len(past_data) != 168:
        raise ValueError("이전 168개의 데이터를 찾을 수 없습니다.")

    input_data_values = past_data.drop('datetime', axis=1).values.astype(np.float32)
    scaled_input_data = scaler_feature[0].transform(input_data_values) # 올바른 scaler 사용

    input_tensor = torch.tensor(scaled_input_data).unsqueeze(0).to(device)

    with torch.no_grad():
      predicted = model(input_tensor)

    predicted = predicted.cpu().numpy()
    predicted_original = scaler_target[1].inverse_transform(predicted) # 올바른 scaler 사용

    pred_arr = make_return_form(predicted_original[0], start_time)
    optiflow_lstm = optimize_pump_flow(pred_arr, predicted_original[0], input_data.waterLevel, 2000)
    result = {"prediction": pred_arr, 'optiflow': optiflow_lstm}
    return result

  except FileNotFoundError:
    raise HTTPException(status_code=404, detail=f"Model or data file for '{input_data.name}' not found.")
  except KeyError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except IndexError as e:
    raise HTTPException(status_code=500, detail=f"Data indexing error: {e}")
  except Exception as e:
    logger.exception(f"Unexpected error in predict_lstm: {e}")
    raise HTTPException(status_code=500, detail="An unexpected error occurred.")




@app.post("/api/predict/xgb")
async def predict_data_xgb(input_data: InputData, model: xgb.Booster = Depends(get_xgb_model), data: pd.DataFrame = Depends(get_xgb_data)):
  try:
    past_data = data[data['datetime'] < input_data.datetime].tail(168)
    start_time = data.iloc[-1]['datetime'] + timedelta(hours=1)

    if len(past_data) != 168:
      raise ValueError("이전 168개의 데이터를 찾을 수 없습니다.")

    input_data_value = past_data.drop('datetime', axis=1).values.astype(np.float32)
    # print(input_data.shape)
    dtest = xgb.DMatrix(input_data_value.reshape(1, -1))
    pred = model.predict(dtest)
    # print(pred)
    result_xgb = make_return_form(pred[0], start_time)

    optiflow_xgb = optimize_pump_flow(result_xgb, pred[0], input_data.waterLevel, 2000)

    result = {"prediction" : result_xgb, 'optiflow' : optiflow_xgb}
    return result

  except FileNotFoundError:
    raise HTTPException(status_code=404, detail=f"Model or data file for '{input_data.name}' not found.")  # 404 Not Found
  except KeyError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")  # 400 Bad Request (키 에러)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))  # 400 Bad Request (잘못된 값)
  except IndexError as e:
    raise HTTPException(status_code=500, detail=f"Data indexing error: {e}")  # 500 Internal Server Error
  except Exception as e:
    logger.exception(f"Unexpected error in predict_lstm: {e}") # 예상 못한 에러 로깅
    raise HTTPException(status_code=500, detail="An unexpected error occurred.") # 일반적인 500 에러.





@app.post("/api/cost")
async def calculate_cost_endpoint(request: Request, input_data: InputData):
  logger.info(f"Request: {request.method} {request.url}")
  model, data = get_xgb_model_and_data(input_data.name)
  try:
    prediction_xgb, outflow_xgb = predict_xgb(input_data.datetime, model, data)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

  id = {'d' : 4, 'j' : 1, 'l' : 11}

  future_data = data_djl[data_djl['observation_time'] >= input_data.datetime & data_djl['reservoir_id'] == id[input_data.name]].head(24)
  start_time = future_data.iloc[0]['observation_time']
  if len(future_data) != 24:
    raise ValueError("이후 24개의 데이터를 찾을 수 없습니다.")

  flow_actual = make_return_form(future_data, start_time)
  cost_actual = calculate_daily_cost_by_linear(flow_actual)
  flow_opti = optimize_pump_flow(prediction_xgb, outflow_xgb, input_data.waterLevel, 2000)
  cost_opti = calculate_daily_cost_by_linear(flow_opti)

  cost_actual_result = make_return_form(cost_actual)
  cost_opti_result = make_return_form(cost_opti)

  result = {'actual' : cost_actual_result, 'optimization' : cost_opti_result}
  return result