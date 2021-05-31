import pandas as pd
from matplotlib import pyplot as plt
import math
import numpy as np

def get_graph(x,y,lx,ly):
    plt.scatter(x,y,color='black')
    plt.plot(lx,ly,color='red')
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.show()

def get_rms(c,m,test_x,test_y):
    tl = len(test_x)
    pred_y=0
    rms=0
    residual=[]
    
    for i in range(tl):
       x = test_x[i]
       y = test_y[i]
       pred_y = c + m*x
       residual.append(y - pred_y) 
       rms = rms + (y - pred_y)**2


    #Residual Plot
    plt.scatter(test_x,residual)
    plt.xlabel("values of x")
    plt.ylabel("residue")
    plt.show()
    
    
    return math.sqrt(rms)/tl
    
    
train = pd.read_csv(r'Lab5\train.csv - train.csv.csv')    
train_x ,train_y = [],[]
for i in range(len(train)):
    train_x.append(train['x'][i])
    train_y.append(train['y'][i])

train_x = pd.array(train_x)
train_y = pd.array(train_y)


test = pd.read_csv(r'Lab5\test.csv - test.csv.csv')
test_x ,test_y = [],[]
for i in range(len(test)):
    test_x.append(test['x'][i])
    test_y.append(test['y'][i])

test_x = pd.array(test_x)
test_y = pd.array(test_y)


###Gradient Descent
m,c=0,0
L = 0.0001
n = float(len(test_x))

for i in range(1000): 
    y_pred = m*test_x + c
    Dm = (-2/n) * sum(test_x * (test_y - y_pred))
    Dc = (-2/n) * sum(test_y - y_pred)
    m = m - L * Dm
    c = c - L * Dc



lx = []
for i in range(len(train)):
    lx.append(train['x'][i])
        
lx = np.array(lx)
ly = lx*m + c

get_graph(test_x,test_y,lx,ly)
print("For m = ", m , "and c = ", c , ",")
print("Root mean square error of model : ",get_rms(c,m,test_x,test_y))
