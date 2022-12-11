import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("fivethirtyeight")
#xs=np.array([1,2,3,4,5],dtype=np.float64)
#ys=np.array([4,4,6,3,8],dtype=np.float64)
def CreateDataset(rng,variance,step=2,correlation=False):
	val=1
	ys=[]
	for i in range(rng):
		y=val+random.randrange(-variance,variance)
		ys.append(y)
		if correlation:
			val+=step
		else:
			val-=step
	xs=[i for i in range(rng)]
	return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def BestFitSlopeAndIntercept(xs,ys):
	m=((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)*mean(xs))-mean(xs*xs))
	b=mean(ys)-m*mean(xs)
	return m,b
def SquaredError(y_orgi,y_line):
	return sum((y_orgi-y_line)**2)
	
def CofficientOfDetermination(y_orgi,y_line):
	y_mean_line=[mean(y_orgi) for y in y_orgi]
	squared_err_mean=SquaredError(y_orgi,y_mean_line)
	squared_regr=SquaredError(y_orgi,y_line)
	return 1-(squared_regr/squared_err_mean)

xs,ys=CreateDataset(40,40,2,correlation="True")
m,b=BestFitSlopeAndIntercept(xs,ys)
regression_line=[(m*x)+b for x in xs]

r_squared=CofficientOfDetermination(ys,regression_line)	
print(r_squared)
predict_x=[41,42]
predict_y=[(m*x)+b for x in predict_x]
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color="g")
plt.plot(xs,regression_line)
plt.show()

print(m,b)