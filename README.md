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
![image](https://user-images.githubusercontent.com/94778101/204469854-685cc4a3-8be3-418d-9a4c-d6c4c9ebc3c7.png)

![image](https://user-images.githubusercontent.com/94778101/204469902-e17639b2-2bf2-4e6c-951d-6d966d9123fa.png)

![image](https://user-images.githubusercontent.com/94778101/204469982-5fa0d0fc-2608-442f-be4d-8aa7b64255ff.png)

![image](https://user-images.githubusercontent.com/94778101/204470028-ce2e13ff-e33a-490e-9db8-aca7aefa3a64.png)

![image](https://user-images.githubusercontent.com/94778101/204470098-f4e44f57-bee7-413c-9331-f2e45993c9ff.png)

![image](https://user-images.githubusercontent.com/94778101/204470203-da32de1c-a2f9-4947-ad95-97defa747447.png)

![image](https://user-images.githubusercontent.com/94778101/204470267-e3c2e4ad-6dcf-427a-85a2-d112362a58d9.png)

![image](https://user-images.githubusercontent.com/94778101/204470326-a9131949-219e-4586-aad0-2881f44fbbe4.png)

![image](https://user-images.githubusercontent.com/94778101/204470392-9edcdc6d-c95a-4369-b505-52384c1a43ab.png)

![image](https://user-images.githubusercontent.com/94778101/204470473-f9adb667-899a-4b80-83e2-0aa0399567f4.png)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

