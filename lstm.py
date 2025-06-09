
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, Lion, Adamax, RMSprop, Nadam, Ftrl, Adadelta, Adagrad, LossScaleOptimizer
from tensorflow.keras import backend as K


#This is the good one: 99.97% Accuracy
from tensorflow.keras.layers import Bidirectional
ticker = 'SPY' #'AAPL'
data = yf.download(ticker, start="2015-01-28", end="2025-04-14",auto_adjust=False)#, interval="1wk")

values = data['Adj Close'].rolling(window=5).mean().dropna().values.reshape(-1, 1) #try mean () .max and .min or others ----
#values = data['Adj Close'].values.reshape(-1,1)


n_steps = 25 # Best at 25 so far
X, y = [], []
for i in range(n_steps, len(values)):
    X.append(values[i - n_steps:i, 0])
    y.append(values[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))


split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale data separately to prevent leakage
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))


X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
y_test = scaler_y.transform(y_test.reshape(-1, 1))

#  Change in Sign 5-Day Moving Average
#                           * 72.459%: {'lstm_units': 41, 'gru_units': 58, 'learning_rate': 0.0005181785280568014, 'batch_size': 16, 'epochs': 59}, 2 LSTM Layers and 1 GRU
model = Sequential()
model.add(LSTM(units=41,                # best units = 41
                return_sequences=True,
                input_shape=(X_train.shape[1], 1),
                activation='tanh',
                recurrent_activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
           #     dropout=0.01,
                recurrent_dropout=0.0,
                seed=None,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                use_cudnn="auto"))

model.add(LSTM(units=41,               # best units = 41
                return_sequences=True,
                activation='tanh',
                recurrent_activation="sigmoid",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
              #  dropout=0.01,
                recurrent_dropout=0.0,
                seed=None,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                use_cudnn="auto"))

model.add(GRU(units=58,
              return_sequences=False,
              activation="tanh",
              recurrent_activation="sigmoid",
              use_bias=True,
              kernel_initializer="glorot_uniform",
              recurrent_initializer="orthogonal",
              bias_initializer="zeros",
              kernel_regularizer=None,
              recurrent_regularizer=None,
              bias_regularizer=None,
              activity_regularizer=None,
              kernel_constraint=None,
              recurrent_constraint=None,
              bias_constraint=None,
              dropout=0.0,
              recurrent_dropout=0.0,
              seed=None,
              return_state=False,
              go_backwards=False,
              stateful=False,
              unroll=False,
              reset_after=True,
              use_cudnn="auto")) # best units = 58



model.add(Dense(units=1,
                activation='linear'))


model.compile(optimizer=Adam(learning_rate=0.0005181785280568014),
              loss='mean_squared_error') # lr = 0.0005181785280568014


history = model.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=16,  # Optimal Batch Size = 16
                    validation_data=(X_test, y_test))


predictions = model.predict(X_test)

predictions = scaler_y.inverse_transform(predictions)
y_test_original =scaler_y.inverse_transform(y_test)


plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('LSTM + GRU Model Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


print("Model Performance:")
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")


x = predictions  # Independent variable (e.g., predictor)
y = y_test_original  # Dependent variable (e.g., target)

x = x.ravel()
y = y.ravel()

data = pd.DataFrame({'x': x, 'y': y})

X = sm.add_constant(data['x'])  # Adds an intercept term
y = data['y']

model = sm.OLS(y, X).fit()

print(model.summary())

comparison_df = pd.DataFrame({
  #  'Open':  yf.download(ticker, start="2005-01-28", end="2025-02-11")['Open'][len(yf.download(ticker, start="2005-01-28", end="2025-02-11")) - len(y_test_original):].values.ravel(),
    'Actual': y_test_original.flatten(),
    'Predicted': predictions.flatten(),
    'Difference': abs(predictions.flatten() - y_test_original.flatten()),
   # '%Change Pred': (np.diff(predictions.flatten())),
   # '%Change Actual': (np.diff(y_test_original.flatten()))

   # 'Difference': abs(predictions.flatten() - y_test_original.flatten()),


})

#comparison_df['%Change Pred'] = (comparison_df['Predicted'].diff() / comparison_df['Predicted'].shift(1))
#comparison_df['%Change Actual'] = (comparison_df['Actual'].diff() / comparison_df['Actual'].shift(1))


df = comparison_df#.sort_values(by='Difference', ascending=False)

# 0.00012286


y_test_new = y_test_original.flatten()
predictions_new = predictions.flatten()

actual_diff = np.diff(y_test_new)
predicted_diff = np.diff(predictions_new)

correct_signs = np.sign(actual_diff) == np.sign(predicted_diff)

accuracy = np.mean(correct_signs)
print()
print()
print(f"Accuracy based on correct trend predictions: {accuracy*100:.4f}%")
print()
print()
print('*'*80)
