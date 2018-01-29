import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

data = np.array(np.load('linRegData.npy'))
data = np.insert(data,0,1,axis=1)

Xplt=data[:,1]
X=data[:,:2]
Y=data[::,2]

#print(data)
print(X.shape)
print(Y.shape)

theta=[0,0]
mse = np.dot(((Y - np.dot(X,theta)).T),(Y - np.dot(X,theta)))
print(mse)

theta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
print(theta)

Ynew = np.dot(theta,X.T)

plt.scatter(Xplt, Y, color='blue')
plt.plot(Xplt, Ynew, color='red')
plt.savefig('linRegData.png')





