# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
Define a function for costFunction,cost and gradient.
2. Load the dataset.
3. Define X and Y array.
4. Plot the decision boundary and Predict the Regression value.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sarankumar J
RegisterNumber: 212221230087

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J
def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```
## Output:
![op1](https://user-images.githubusercontent.com/93427303/196707079-8d7bf851-194c-452f-bfed-5396e9dbad54.png)

![op2](https://user-images.githubusercontent.com/93427303/196707095-570972a4-60e0-4563-b5fc-23f3ea136db3.png)

![op3](https://user-images.githubusercontent.com/93427303/196707123-a383679b-f9b6-454b-9195-db73c847c0e8.png)

![op4](https://user-images.githubusercontent.com/93427303/196707147-3793da20-2df7-4ae1-b167-a75b8425ab21.png)

![op5](https://user-images.githubusercontent.com/93427303/196707161-7a05926b-ab4f-42e0-bebb-d3bbecb5f5c2.png)

![op6](https://user-images.githubusercontent.com/93427303/196707183-246bef46-ebe0-43c6-b2f2-454ae9917223.png)

![op7](https://user-images.githubusercontent.com/93427303/196707212-82fe0851-6fcb-4cfc-b529-9e8202e42eab.png)

![op8](https://user-images.githubusercontent.com/93427303/196707240-4f9021b0-ebb7-4de8-aee9-066a1ab1a1d4.png)

![op9](https://user-images.githubusercontent.com/93427303/196707267-cb3e71ea-b21a-493a-ab92-fc02ef48f92d.png)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

