import pandas as pd  
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
#print("loading the data")
df = pd.read_csv("Stock_prize_prediction/NVDA.csv")
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Nvidia Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

#print(df.isnull().sum())
#print("There are no missing null values")

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']


def compute_rsi(data, window):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

short_ema = df['Close'].ewm(span=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = short_ema - long_ema
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
df['SMA'] = df['Close'].rolling(window=20).mean()
df['RSI'] = compute_rsi(df['Close'],14)
df.dropna(inplace=True)
features = ['Close', 'SMA','RSI', 'MACD','Signal_Line','MACD_Hist' ] 
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

X = df_scaled[:-1]
y = (df['Close'].shift(-1) > df['Close']).astype(int).values[:-1]
x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, shuffle=False)

models = [LogisticRegression(), SVC(
  kernel='poly', probability=True),]

for i in range(2):
  models[i].fit(x_train, y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    y_train, models[i].predict_proba(x_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    y_test, models[i].predict_proba(x_test)[:,1]))
  print()

y_regression = df_scaled[1:, 0]
_, _, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, shuffle=False)
lr_model = LinearRegression()
lr_model.fit(x_train, y_train_reg)
y_pred_train = lr_model.predict(x_train)
y_pred_test = lr_model.predict(x_test)

print('Linear Regression (Regression):')
print('  Training RMSE:', np.sqrt(mean_squared_error(y_train_reg, y_pred_train)))
print('  Validation RMSE:', np.sqrt(mean_squared_error(y_test_reg, y_pred_test)))
print('  Training R²:', r2_score(y_train_reg, y_pred_train))
print('  Validation R²:', r2_score(y_test_reg, y_pred_test))

print("as we know that linear regression is best fit model")
print("hence")
y_test = y_test.flatten()
y_pred_test = y_pred_test.flatten()

# Optional: Convert back to real prices (denormalized)
# Only do this if you want actual price in dollars, otherwise skip this block
# Here we assume the first column is 'Close' (the one you predicted)
y_test_actual = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), df_scaled.shape[1]-1))], axis=1)
)[:,0]

y_pred_actual = scaler.inverse_transform(
    np.concatenate([y_pred_test.reshape(-1,1), np.zeros((len(y_pred_test), df_scaled.shape[1]-1))], axis=1)
)[:,0]

# Fix Date Index
df['Date'] = pd.to_datetime(df['Date'])      # Convert to datetime if not already
df.set_index('Date', inplace=True)

# Get date range for the test set
test_dates = df.index[-len(y_test):]

# Plot
plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test_actual, label='Actual Price', linewidth=2)
plt.plot(test_dates, y_pred_actual, label='Predicted Price', color='orange')
plt.title('Nvidia Stock Price Prediction (Test Set)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()