import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from datetime import datetime, timedelta

coin_name = "ethereum" # maybe ethereum/ripple/titecoin...
# start_day = "20130428" # the day established bitcoin
end_day = time.strftime("%Y%m%d") #today
start_day = (datetime.today() - timedelta(days=1000)).strftime("%Y%m%d")
# print start_day
# print end_day

# pre-process data
print ("Start load Data...")

# get market info for bitcoin from the start day to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/"+coin_name+"/historical-data/?start="+start_day+"&end="+end_day)[0]
# convert the date string to the correct date format
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
# when Volume is equal to '-' convert it to 0
bitcoin_market_info.loc[bitcoin_market_info['Volume'].isin(['-']),'Volume']=0
# convert to int
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
# Rename column name
bitcoin_market_info.columns= ["Date", "Open", "High", "Low", "Close", "Volume", "Market_Capitalization"]
# # look at the first few rows
# print bitcoin_market_info.head()

# # plot data to check
# bitcoin_market_info.plot(x = "Date", y="Close", style='-')
# plt.xlabel("Date")
# plt.ylabel("Close Price ($)")
# plt.show()
print ("Load data finished!!!")

print ("---------------------")

# LSTM predict for close price
print ("Start Training...")

training_set = bitcoin_market_info.iloc[:,4:5]
# print training_set.head()

# Converting into 2D array
training_set = training_set.values
# print training_set.shape

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
# print training_set

X_train = training_set[0:799]
Y_train = training_set[1:800]

#Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
X_train = np.reshape(X_train, (799, 1, 1))
X_train

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model = Sequential()

#Adding the input layer and the LSTM layer
model.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
#Adding the output layer
model.add(Dense(units = 1))
#Compiling the Recurrent Neural Network
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Fitting the Recurrent Neural Network [epoches is a kindoff number of iteration]
model.fit(X_train, Y_train, batch_size = 32, epochs = 200)
model.save("model_"+coin_name+".h5")
print ("Training LSTM finished!!!")