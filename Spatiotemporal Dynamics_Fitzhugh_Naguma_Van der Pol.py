import pylab as plt
from random import random
import math
import numpy.matlib
from numpy.random import randn
import numpy as np
import scipy.integrate as integrate
from matplotlib import cm



tc=10
X0 =[-1, 1]; # initial state for truth
c=0.3
v=2.57+0.35
tspan=np.arange(0.01,tc+0.01,0.01) #timestep
x=np.arange(0,60+0.01,0.01)
aa=np.sqrt(1-(c/v)**2)

from pattern_func import Vanderpol2 # THIS IS THE COMMAND, FILE NAME IS van and Funcyion name os VanderPOl2
from scipy.integrate import odeint

h=np.zeros((len(x),len(tspan)))
for i in range(len(tspan)):
    zz=((x/v)+tspan[i])/aa
    xr=odeint(Vanderpol2,X0,zz)
    h[:,i]=xr[:,0]

plt.plot(x,h[:,-1])
#for i in range(len(tspan)):
#    cx=[i + random() for i in range(len(x))]
#    plt.scatter(np.matlib.repmat(tspan[i],len(x),1),h[:,i])
##    plt.plot(np.matlib.repmat(tspan[i],len(x),1), h[:,i])#, c=cx)#, cmap=cm.Greys)
#    plt.show()
    
#xx,yy = np.meshgrid(tspan,h[:, 0])
#plt.pcolormesh(xx, yy, h)
###    
#plt.imshow(h)
###    
#import matplotlib.image as mpimg 
#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
#
#img = mpimg.imread('lena.png')
#gray = rgb2gray(img)
#plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.savefig('lena_greyscale.png')
#plt.show()
   
