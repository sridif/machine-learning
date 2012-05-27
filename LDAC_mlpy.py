# liner discriminant analysis done as per example found below...
# http://mlpy.sourceforge.net/docs/3.5/lin_class.html#linear-discriminant-analysis-classifier-ldac


import numpy as np
import matplotlib.pyplot as plt
import mlpy

np.random.seed(0)

mean1, cov1, n1 = [1, 5], [[1, 1],[1, 2]], 200
x1 = np.random.multivariate_normal(mean1, cov1, n1)
y1 = np.ones(n1,dtype= np.int)

mean2, cov2, n2 = [2.5, 2.5], [[1, 0],[0, 1]], 300
x2 = np.random.multivariate_normal(mean2, cov2, n2)
y2 = -np.ones(n2,dtype= np.int)

x= np.concatenate((x1,x2), axis=0 )
y= np.concatenate((y1,y2))
ldac= mlpy.LDAC()
ldac.learn(x,y)
w=ldac.w()
b=ldac.bias()
xx= np.arange(np.min(x[:,0]), np.max(x[:,0]), 0.01)
yy= - (w[0]* xx + b) / w[1] # seperator line 

fig= plt.figure(1)
plot1= plt.plot(x1[:,0], x1[:,1], 'ob', x2[:, 0], x2[:,1], 'or')
plot2= plt.plot(xx, yy, '--k')
plt.show()



