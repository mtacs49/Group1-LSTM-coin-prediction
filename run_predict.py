import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
import datetime
import numpy as np
from datetime import datetime, timedelta

# Reading CSV file from test set
coin_name = "ethereum" # maybe ethereum/ripple/titecoin...
# start_day = "20130428" # the day established bitcoin
end_day = time.strftime("%Y%m%d") #today
start_day = (datetime.today() - timedelta(days=50)).strftime("%Y%m%d")
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

real_coin_price = bitcoin_market_info.iloc[:,4:5]
# print training_set.head()
# print len(real_coin_price)

# Converting into 2D array
real_coin_price = real_coin_price.values

# Scaling of Data [Normalization]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
real_coin_price = sc.fit_transform(real_coin_price)

#getting the predicted BTC value of the first week of Dec 2017  
inputs = real_coin_price
inputs = sc.fit_transform(inputs)
inputs = sc.transform(inputs)

#Reshaping for Keras [reshape into 3 dimensions, [batch_size, timesteps, input_dim]
inputs = np.reshape(inputs, (50, 1, 1))

# load model
from keras.models import Sequential
from keras.models import load_model
model = load_model("model_"+coin_name+".h5")

predicted_coin_price = model.predict(inputs)
predicted_coin_price = sc.inverse_transform(predicted_coin_price)

#Graphs for predicted values
plt.plot(real_coin_price, color = 'red', label = 'Real coin Value')
plt.plot(predicted_coin_price, color = 'blue', label = 'Predicted coin Value')
plt.title('Coin Value Prediction')
plt.xlabel('Days')
plt.ylabel('Coin Value')
plt.legend()
plt.show()