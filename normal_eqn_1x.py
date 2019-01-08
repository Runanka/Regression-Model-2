import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("This script takes data.csv as default data")

x_label=input("Enter x label: ")
y_label=input("Enter y label: ")

data=pd.read_csv("data.csv", header=None)

m=len(data)
x=np.hstack((data.iloc[:, 0])).reshape(m,1)
Y=np.hstack((data.iloc[:, 1])).reshape(m,1)

ones= np.ones((m,1))
X=np.hstack((ones,x))

def costfn(X, Y, theta):
    temp=X.dot(theta)-Y
    return np.sum(np.power(temp,2))/(2*m)

def normal(X, Y):
    temp=X.transpose()
    temp=temp.dot(X)
    temp=np.linalg.inv(temp)
    temp=temp.dot(X.transpose())
    theta=temp.dot(Y)
    return theta

theta=normal(X, Y)
cost_value=costfn(X, Y, theta)

print("Value of theta and minimum error")
print(theta)
print(cost_value)

plt.scatter(x, Y)
plt.plot(x, X.dot(theta), 'r')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.show()

min_theta=int(np.amin(theta)-5)
max_theta=int(np.amax(theta)+5)

theta0=np.arange(min_theta,max_theta,0.1)
theta1=np.arange(min_theta,max_theta,0.1)
j_vals= np.zeros((len(theta0),len(theta1)))

t=np.ones((2,1))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        t[0][0]=theta0[i]
        t[1][0]=theta1[j]
        j_vals[i][j]=costfn(X, Y, t)

plt.contourf(theta0,theta1, j_vals)
plt.show()