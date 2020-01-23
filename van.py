import pylab as plt
from numpy.random import randn
import math
import numpy as np

def Vanderpol2(state,t):
    x,y,z = state   
    dx=z*(y-(1/3)*x**3+x)
    dy=-(1/z)*(x-1.35*math.cos(t))
    dz=0
    return dx,dy,dz

def Vanderpolt(state,t):
    x,y,z = state   
    dx=z*(y-(1/3)*x**3+x)
    dy=-(1/z)*(x-1.35*math.cos(t))
    dz=0
    return dx,dy,dz

def particle(cum,mem):
    j=1
    y=(1/mem)*np.random.uniform(0,1,1)
    for i in range(mem):
        u=y+(1/mem)*(i-1)
        while u>cum[j]:
            j=j+1
            i+=1
        return j
        