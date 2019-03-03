# Import Required Libaries
import numpy as np
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pandas_datareader import data as pdr

# Here we override the default datareader function
# Since Yahoo Finance has decommissioned their historical data API
import fix_yahoo_finance as yf
yf.pdr_override()

# Import Data
df = pdr.get_data_yahoo('^HSI') # Get Historical Finance Data of Hang Seng Index 
df = df.dropna() # Remove missing values from the data
df = df.iloc[:, :4] # Only Open, High, Low, Close data are used 

#
# Predictor / Independent Variables 
df['S_10'] = df['Close'].rolling(window=10).mean() # 10-days moving average
df['Corr'] = df['Close'].rolling(window=10).corr(df['S_10']) # Correlation
df['RSI'] = ta.RSI(np.array(df['Close']), timeperiod =10) # Relative Strength Index
df['Open-Close'] = df['Open'] - df['Close'].shift(1) # Different between close price of yesterday and open rice today
df['Open-Open'] = df['Open'] - df['Open'].shift(1) # Different between open price of yesterday and today
df = df.dropna()
X = df.iloc[:,:9]

# 
# Target / Dependent Variable
# Buy   [1]: Tomorrow closing is higher than today's closing
# Sell [-1]: Else
y = np.where (df['Close'].shift(-1) > df['Close'],1,-1)


# 
# Splitting the dataset
# 70% Train - 30% Test
split = int(0.7*len(df))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:] # Python List Comprehension

# Instantiate Logistic Regression
model = LogisticRegression()
model = model.fit(X_train, y_train)

# Examine the coefficients
temp = list(zip(X.columns, np.transpose(model.coef_)))
print(pd.DataFrame(temp))

# Class Probabilities
probability = model.predict_proba(X_test) # Probability Estimates
print("Probability:",probability)

# Predicted Class Labels
# 1 if second column in probability is > 0.5
# else -1 
predicted = model.predict(X_test) 


# Evaluation
# Confusion Matrix
# Perfomance of classification model for which the true values are known
print("#Confusion Matrix#")
print(metrics.confusion_matrix(y_test, predicted))

# Classification Report
# f1-score: accuracy in classifying the data points in that particular class compared to all other class
# Support: number of samples of the true response that lies in that class
print("#Classification Report#")
print(metrics.classification_report(y_test, predicted))

# Model Accuracy
print("#Accuracy#")
print(model.score(X_test,y_test))


# Cross Validation
# Cross check accuracy of the model using 10-fold cross-validation
cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)

print(cross_val)
print(cross_val.mean())

# Trading Strategy 
# Cumulative Nifty 50 returns 
df['Nifty_returns'] = np.log(df['Close']/df['Close'].shift(1))
Cumulative_Nifty_returns = np.cumsum(df[split:]['Nifty_returns'])
# Signal predicted by the model ( Buy 1/ Sell -1 )
df['Predicted_Signal'] = model.predict(X)
df['Startegy_returns'] = df['Nifty_returns']* df['Predicted_Signal'].shift(1)
Cumulative_Strategy_returns = np.cumsum(df[split:]['Startegy_returns'])

# Plotting
plt.figure(figsize=(10,5))
plt.plot(Cumulative_Nifty_returns, color='r',label = 'Nifty Returns')
plt.plot(Cumulative_Strategy_returns, color='g', label = 'Strategy Returns')
plt.legend()
plt.show()