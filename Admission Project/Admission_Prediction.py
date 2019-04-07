import numpy as np
#import numpy.linalg as inv
import pandas as pd
import matplotlib.pyplot as plt
def normalization(x):
    max=np.amax(x)
    min=np.amin(x)
    mean=np.mean(x)
    for i in range(len(x)):
        x[i,0]=((x[i,0]-mean)/(max-min))
    return x
##############################################
def costError(X,Y,theta): 
    return (1/(2*len(X)))*np.sum(np.power(((X*theta)-Y),2))

##############################################
def gridentDecent(X,Y,theta,alpha,iters):
    m=X.shape[1]
    temp=np.zeros((m,1))
    for i in range(iters):
        X=normalization(X)
        hx=X*theta
        for j in range(m):
            dot=np.multiply((hx-Y),X[:,j:j+1])
            temp[j,0]=theta[j,0]-((alpha/m)*np.sum(dot))
        theta=temp
        
    return theta
#################################################################### 
def normalEquation(X,Y):
    theta=np.linalg.inv(X.T*X)*X.T*Y
    return theta
###################################################################   
path="Admission_Predict_Ver1.1.csv"
datarp=pd.read_csv(path,header=None)
datarp.insert(1,'Ones',1)
#print(datarp.head(10))
X=np.matrix(datarp.iloc[1:,1:9])
X=X.astype(np.float)
x=np.array(datarp.iloc[1:,2:3])
Y=np.matrix(datarp.iloc[1:,9:10])

print()
Y=Y.astype(np.float)
y=np.array(datarp.iloc[1:,9:10])
y=y.astype(np.float)
theta=np.zeros((8,1))
theta=normalEquation(X,Y)
hx=X[:,:]*theta[:,:]
x=x.astype(np.float)
plt.scatter(x,y)
plt.plot(X[:,1:2],hx,'r')
#Xtest=np.matrix(datarp.iloc[10:15,1:9])
#Xtest=Xtest.astype(np.float)
Xtest=np.matrix([[1,325,112,4,4,4,9,1]])

