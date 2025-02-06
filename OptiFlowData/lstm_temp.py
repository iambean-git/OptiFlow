from preamble import *
import holidays
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

df = pd.read_csv('./data/j_weather_data_v2.csv', index_col=0)
 
feature = df.reset_index()
target = df['outflow'].values.reshape(-1, 1)
feature['datetime'] = pd.to_datetime(feature['datetime'])

kr_holidays = holidays.KR()
feature['is_weekend'] = feature['datetime'].dt.dayofweek >= 5 | feature['datetime'].isin(kr_holidays)
 
train_size = int(feature.shape[0] * 0.7)

trainset_feature = feature[:train_size]
trainset_target = target[:train_size]
testset_feature = feature[train_size:]
testset_target = target[train_size:]

def create_time_feature(df): 
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
  return df
 
trainset_feature = create_time_feature(trainset_feature)
testset_feature = create_time_feature(testset_feature)
 
trainset_feature.drop('datetime', axis=1, inplace=True)
testset_feature.drop('datetime', axis=1, inplace=True)

scaler_feature = MinMaxScaler()
train_feature_scaled = scaler_feature.fit_transform(trainset_feature)
test_feature_scaled = scaler_feature.transform(testset_feature)

scaler_target = MinMaxScaler()
train_target_scaled = scaler_target.fit_transform(trainset_target)
test_target_scaled = scaler_target.transform(testset_target)

def split_dataset(data, target, seq_len):
  X, y  = [], []
  for i in tqdm(range(data.shape[0]-(seq_len))):
    X.append(data[i:i+seq_len, :]) 
    y.append(target[i+seq_len])
  return np.array(X), np.array(y)

seq_len = 60
 
X_train, y_train = split_dataset(train_feature_scaled, train_target_scaled, seq_len)
X_test, y_test = split_dataset(test_feature_scaled, test_target_scaled, seq_len)

X_train_tensor, y_train_tensor = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test_tensor, y_test_tensor = torch.tensor(X_test).float(), torch.tensor(y_test).float()

loader_train = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), shuffle=True,
                                           batch_size=128)
loader_test = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), shuffle=False,
                                          batch_size=128)
 
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
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)
 
for I, label in loader_train:
  print(model(I)[0], label[0])
  break
 
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
num_epochs = 200
model.train()
model.to(device)
for epoch in range(num_epochs):
  running_loss = 0.0
  for X_batch, y_batch in loader_train:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    out = model(X_batch)
    loss = loss_fn(out, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(loader_train.dataset):.8f}")
 
model.eval()

y_pred_list = []
with torch.no_grad():
  for X_batch, _ in loader_test:
    X_batch = X_batch.to(device)
    y_pred = model(X_batch).cpu().numpy()
    y_pred_list.append(y_pred)

y_pred = np.concatenate(y_pred_list, axis=0)
 
from sklearn.metrics import mean_squared_error, r2_score

y_test_ori = scaler_target.inverse_transform(y_test)
y_pred_ori = scaler_target.inverse_transform(y_pred)

mse = mean_squared_error(y_test, y_pred)
mse_ori = mean_squared_error(y_test_ori, y_pred_ori)
rmse = np.sqrt(mse)
rmse_ori = np.sqrt(mse_ori)
r2 = r2_score(y_test, y_pred)
print(f'RMSE : {rmse}, RMSE_ORI : {rmse_ori} R-value : {r2}')