import pandas as pd
import quandl,datetime
import matplotlib.pyplot as plt
from matplotlib import style
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression
from joblib import dump,load


style.use('ggplot')
df=quandl.get("WIKI/GOOGL")
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT']=((df['Adj. High']-df['Adj. Close'])/df['Adj. Close'])*100
df['PCT_Change']=((df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'])*100
df=df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
forcasted_col='Adj. Close'
df.fillna(-9999,inplace=True)
forcasted_out=int(math.ceil(0.01*len(df)))
df['label']=df[forcasted_col].shift(-forcasted_out)
df.dropna(inplace=True)
x=np.array(df.drop(["label"],axis=1))
y=np.array(df["label"])
y=y[:-forcasted_out]
x=preprocessing.scale(x)
x=x[:-forcasted_out]
x_late=x[-forcasted_out:]
df.dropna(inplace=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=LinearRegression(n_jobs=-1)
clf.fit(x_train,y_train)
dump(clf,"clf.joblib")
clf=load("clf.joblib")
accuracy=clf.score(x_test,y_test)
predicted=clf.predict(x_late)


df['Forecast']=np.nan
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day
for i in predicted:
	next_date=datetime.datetime.fromtimestamp(next_unix)
	next_unix+=one_day
	df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] +[i]


#df['Adj. Close'].plot()
#df['Forecast'].plot()
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.show()