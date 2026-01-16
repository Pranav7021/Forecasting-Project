# importing relevant packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.callbacks as C
import traceback
from time import time
import ta

# Data can be found at https://www.kaggle.com/datasets/kalilurrahman/coca-cola-stock-live-and-updated/
# Loading dataset into a pandas dataframe
# The data is Coca Cola stock data, because there is a lot of data for this stock and it provides slow and consistent growth.
df = pd.read_csv("/content/drive/MyDrive/Coca-Cola_stock_history.csv")

# Examining the data set
df.head()

print((df['Date'].isnull()).sum()) # will replace
print((df['Open'].isnull()).sum()) # will remove
print((df['High'].isnull()).sum()) # will remove
print((df['Low'].isnull()).sum()) # will remove
print((df['Close'].isnull()).sum()) # will keep
print((df['Volume'].isnull()).sum()) # will keep
print((df['Stock Splits'].isnull()).sum()) # will use to adjust close prices

# Removing unnecessary features
del df['Open']
del df['High']
del df['Low']
del df['Dividends']
del df['Stock Splits']

'''
Preparing the technical indicators for use in the data. To obtain the technical indicators from the data, I will use the TA-LIB library.
Documentation can be found at https://technical-analysis-library-in-python.readthedocs.io/_/downloads/en/latest/pdf/

The technical indicators that I will use are:

Momentum:
- KAMA (ta.momentum.KAMAIndicator())
- ROC (ta.momentum.ROCIndicator())
- tsi (ta.momentum.TSIIndicator())

Volatility:
- Bollinger Bands (ta.volatility.BollingerBands())

Trend:
- Aroon (ta.trend.AroonIndicator())
- DPO (ta.trend.DPOIndicator())
- EMA (ta.trend.EMAIndicator())
- MACD (ta.trend.MACD())
- TRIX (ta.trend.TRIXIndicator())
'''

kama1 = ta.momentum.KAMAIndicator(close = df['Close'], window = 10, pow1 = 2, pow2 = 30, fillna = False)
df['KAMA'] = kama1.kama()

roc1 = ta.momentum.ROCIndicator(close = df['Close'], window = 12, fillna = False)
df['ROC'] = roc1.roc()

tsi1 = ta.momentum.TSIIndicator(close = df['Close'], window_slow = 25, window_fast = 13, fillna = False)
df['TSI'] = tsi1.tsi()

bb1 = ta.volatility.BollingerBands(close = df['Close'], window = 20, window_dev = 2, fillna = False)
df['bb_h'] = bb1.bollinger_hband()
df['bb_hi'] = bb1.bollinger_hband_indicator()
df['bb_l'] = bb1.bollinger_lband()
df['bb_li'] = bb1.bollinger_lband_indicator()
df['bb_ma'] = bb1.bollinger_mavg()
df['bb_pband'] = bb1.bollinger_pband()
df['bb_wband'] = bb1.bollinger_wband()

aroon1 = ta.trend.AroonIndicator(df['Close'], df['Close'], window = 25, fillna = False)
df['aroon_down'] = aroon1.aroon_down()
df['aroon_indicator'] = aroon1.aroon_indicator()
df['aroon_up'] = aroon1.aroon_up()

dpo1 = ta.trend.DPOIndicator(close = df['Close'], window = 20, fillna = False)
df['DPO'] = dpo1.dpo()

ema1 = ta.trend.EMAIndicator(close = df['Close'], window = 14, fillna = False)
df['EMA'] = ema1.ema_indicator()

macd1 = ta.trend.MACD(close = df['Close'], window_slow = 26, window_fast = 12, window_sign = 9, fillna = False)
df['MACD_diff'] = macd1.macd_diff()

trix1 = ta.trend.TRIXIndicator(close = df['Close'], window = 15, fillna = False)
df['TRIX'] = trix1.trix()

df = df.iloc[43:]

# Removing more irrelevant features

del df['Date']
del df['bb_hi']
del df['bb_li']
del df['bb_pband']
del df['bb_wband']
del df['bb_ma']
del df['aroon_down']
del df['aroon_up']
del df['Volume']
del df['DPO']
del df['ROC']
del df['EMA']
del df['TSI']

df.head(5)

# Test-train split
df_train = df[9000:11950]
df_test = df[11950:]
print(df_train.shape)
print(df_test.shape)
print(df_train.columns)

# Normalizing training data

df_train_scaled = df_train.copy()
means = []
stds = []

# Apply normalization techniques
for column in df_train_scaled.columns:
  means.append(df_train_scaled[column].mean())
  stds.append(df_train_scaled[column].std())
  df_train_scaled[column] = (df_train_scaled[column] -
                           df_train_scaled[column].mean()) / df_train_scaled[column].std()

# View normalized data
df_train_scaled.head(10)

# Normalizing testing data

df_test_scaled = df_test.copy()
means_test = []
stds_test = []

# Apply normalization techniques
for column in df_test_scaled.columns:
  means_test.append(df_test_scaled[column].mean())
  stds_test.append(df_test_scaled[column].std())
  df_test_scaled[column] = (df_test_scaled[column] -
                           df_test_scaled[column].mean()) / df_test_scaled[column].std()

# View normalized data
df_test_scaled.head(6)

# ----------------- IMPLEMENTATION OF BEDCA MODEL -----------------

# The belief update layer
@tf.keras.saving.register_keras_serializable()
class belief_update_layer(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.convolution_layer = tf.keras.layers.Conv2D( #5, 3
          filters = 8, kernel_size = 4, strides = 2, padding = "valid", data_format = "channels_last", input_shape = (50, 50, 1), use_bias = False
    )

  def call(self, inputs):
    with tf.GradientTape() as tape:
      tape.watch(inputs)
      x1 = tf.reshape(inputs, [1, 50])
      tape.watch(x1)

      x2 = tf.matmul(x1, x1, transpose_a = True, transpose_b = False) #x1 = (50, 50)
      tape.watch(x2)
      x3 = tf.reshape(x2, [1, 50, 50, 1])
      tape.watch(x3)

      x4 = self.convolution_layer(x3) #x1 = (1, 45, 45, 1)
      tape.watch(x4)

    return x4 #shape: (20, 20)

# The Encoder part of the BEDCA model
class Belief_Encoder(tf.keras.Model):
  def __init__(self, features_size, input_size):
    super().__init__()
    self.features_size = features_size
    self.update_layers = []

    for i in range(features_size):
      self.update_layers.append(belief_update_layer())

    self.conv1a = tf.keras.layers.Conv2D(
        filters = 16, kernel_size = 3, padding = "valid", data_format = "channels_last", input_shape = (24, 24, 8)
    )

    self.conv2a = tf.keras.layers.Conv2D(
        filters = 32, kernel_size = 2, strides = 2, padding = "valid", data_format = "channels_last", input_shape = (22, 22, 16), activation = 'relu'
    )

    self.max_pool = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), data_format = 'channels_last', input_shape = (11, 11, 32))

  def call(self, inputs):
    with tf.GradientTape() as tape:
      x = inputs

      x_adj_new = []
      for i in range(7):
        x_adj_new.append(self.update_layers[i](x[i]))

      r = tf.math.reduce_sum(x_adj_new, 0)

      x = self.conv1a(r)

      x = self.conv2a(x)

      x = self.max_pool(x)

    return x #shape: (1, 7, 7, 8)

# The decoder part of the BEDCA model
class Belief_Decoder(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.conv_transpose = tf.keras.layers.Conv2DTranspose(
        filters = 32, kernel_size = 5, padding = "valid", data_format = "channels_last"
    )

    self.flatten_layer1 = tf.keras.layers.Flatten(
        data_format = 'channels_last'
    )

    self.dnn = tf.keras.layers.Dense(
        units = 1024, activation = 'relu'
    ) #input_shape = (1, 392),

    self.dnn2 = tf.keras.layers.Dense(
        units = 256, activation = 'relu'
    )

    self.dnn3 = tf.keras.layers.Dense(
        units = 128, activation = 'relu'
    )

    self.dnn4 = tf.keras.layers.Dense(
        units = 32, activation = 'tanh'
    )

    self.dnn5 = tf.keras.layers.Dense(
        units = 4, activation = 'elu'
    )

    self.dnn6 = tf.keras.layers.Dense(
        units = 2, activation = 'linear'
    )

  def call(self, inputs):
    with tf.GradientTape() as d_tape:
      d_tape.watch(inputs)

      x3 = self.conv_transpose(inputs)
      d_tape.watch(self.conv_transpose.kernel)
      d_tape.watch(self.conv_transpose.bias)
      d_tape.watch(x3)

      x4 = self.flatten_layer1(x3)
      d_tape.watch(x4)

      x5 = self.dnn(x4)
      d_tape.watch(self.dnn.kernel)
      d_tape.watch(self.dnn.bias)
      d_tape.watch(x5)

      x6 = self.dnn2(x5)
      d_tape.watch(self.dnn2.kernel)
      d_tape.watch(self.dnn2.bias)
      d_tape.watch(x6)

      x7 = self.dnn3(x6)
      d_tape.watch(self.dnn3.kernel)
      d_tape.watch(self.dnn3.bias)
      d_tape.watch(x7)

      x8 = self.dnn4(x7)
      d_tape.watch(self.dnn4.kernel)
      d_tape.watch(self.dnn4.bias)
      d_tape.watch(x8)

      x9 = self.dnn5(x8)
      d_tape.watch(self.dnn5.kernel)
      d_tape.watch(self.dnn5.bias)
      d_tape.watch(x9)

      x10 = self.dnn6(x9)
      d_tape.watch(self.dnn6.kernel)
      d_tape.watch(self.dnn6.bias)
      d_tape.watch(x10)

    return x10

# Putting it together, the BEDCA model
class Belief_Model(tf.keras.Model):
  def __init__(self, features_size, input_size):
    super().__init__()
    self.features_size = features_size
    self.input_size = input_size

    self.encoder_arch = Belief_Encoder(features_size, input_size)
    self.decoder_arch = Belief_Decoder()

  def call(self, inputs):
    x4 = inputs
    x4 = self.encoder_arch(x4) #shape: (1, 7, 7, 8)
    x4 = self.decoder_arch(x4) #2 numbers

    return x4[0]

# -------------------- TRAINING THE MODEL --------------------

My_Encoder_Decoder = Belief_Model(7, 50) #151

def loss_function(y_act, y_pred):
  with tf.GradientTape() as tape:
    tape.watch(y_act)
    tape.watch(y_pred)
    custom_loss = (y_pred[0] - y_act[0]) ** 2 + 0.5 * (y_pred[1] - y_act[1]) ** 2
    tape.watch(custom_loss)

  return custom_loss

NUM_EPOCHS = 150
BATCH_SIZE = 1

optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate = 0.0005)

checkpoint_dir = '[custom_location]'
checkpoint_prefix = '[custom_location]'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                My_Encoder_Decoder = My_Encoder_Decoder)

train_losses = []
My_Encoder_Decoder.build((7, 50))

def training_step(encoder_input, real_output):
  with tf.GradientTape() as cur_tape:
    y_pred = My_Encoder_Decoder(encoder_input)

    my_loss = loss_function(real_output, y_pred)

  gradients = cur_tape.gradient(my_loss, My_Encoder_Decoder.trainable_weights)
  optimizer.apply_gradients(zip(gradients, My_Encoder_Decoder.trainable_weights))

  return my_loss

for epoch in range(NUM_EPOCHS):
  cur_loss = 0
  start_time = time()

  for step in range(0, int(TRAIN_INPUT_LENGTH / BATCH_SIZE)):

    encoder_input = batched_input_train_x[step]
    actual_output = tf.convert_to_tensor(train_y[step])

    cur_loss += training_step(encoder_input, actual_output)

  print(My_Encoder_Decoder(batched_input_train_x[0]), tf.convert_to_tensor(train_y[0]), loss_function(tf.convert_to_tensor(train_y[0]), My_Encoder_Decoder( batched_input_train_x[0] )))
  print("Epoch %d: Training Loss %.4f" % (epoch + 1, cur_loss/TRAIN_INPUT_LENGTH))

  train_losses.append(cur_loss/TRAIN_INPUT_LENGTH)

checkpoint.save(file_prefix=checkpoint_prefix)
np.save("[custom_location]", train_losses)
