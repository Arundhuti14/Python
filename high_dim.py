import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
from scipy.integrate import odeint
from functs import Vanderpolt 
from functs import Vanderpol
from functs import Vanderpol2
from functs import show_patterns
from numpy.linalg import inv
T = 0.3# total time
tot=10
dt = 0.5  # time step
X0 =[0.1, 0.1]# initial state for truth
b = -0.92
dx = 0.1  # space step
c=0.3
S=0.1
v=5.42
aa=np.sqrt(1-(c/v)**2)
tspan=np.arange(0,T+0.001,0.1)
x=np.arange(0,S+0.001,dx)
timestep=np.arange(0,tot+0.001,0.1) 
time= np.arange(dt,tot+0.001,dt) 
#for i in range(len(time)):
#    time[i]="{0:0.1f}".format(time[i])

times=range(len(timestep))
size =len(x)  # size of the 2D grid
for i in range(len(tspan)):
    z=((x/v)+tspan[i])/aa
    xr=odeint(Vanderpolt,X0,z)    
U = xr[-1,0] +np.sqrt(0.01)* np.random.randn(size, 1)
V = xr[-1,1] +np.sqrt(0.01)*np.random.randn(size, 1)
y0=np.concatenate((U, V), axis=0)
ys=np.random.randn(size*2)
for i in range(len(ys)):
    ys[i]=y0[i,:]# initial condition

truth=odeint(Vanderpol,ys,timestep) 
   
R=0.001
matrixR=np.diag([R]*size*2)

obs=np.random.randn(len(time),size*2) 
g=0
for i in range(len(time)):
    g=g+5
    obs[i,:]=truth[g,:]+np.random.normal(0, math.sqrt(R),size*2) 

mem=20
sigma=0.01
x0_en=np.random.randn(size*2,mem) 
for i in range(size*2):
    x0_en[i,:]=ys[i]+np.random.normal(0, math.sqrt(sigma), mem)

# particle filter
W=[]    
sss=[]
ts=[]
pxf=[]
pmean=[]
posmean=[]
NEFF=[]
tim=0
for t in time: 
    x_f= np.zeros((mem,size*2))  
    s=0;
    ensem=[];
    for i in np.arange(0,mem):
        tspan2=np.arange(t-dt,t+0.001,0.1)
        XX=x0_en[:,i]
        z=odeint(Vanderpol,XX,tspan2)
        s = s+z
        x_f[i,:]= z[-1,:] 
        ensem.append(z)     
    sss.append(ensem)
    if t==tot:
       ts.append(tspan2)
       mean=s/mem        
    else:
        ts.append(tspan2[0:-1])
        mean=s[0:-1,:]/mem
    
    pxf.append(x_f)           
    pmean.append(mean)       
    vhat=obs[tim,:]-x_f
    
    weight=np.zeros((mem,1))
    for i in range(mem):
        #weight[i]= 1/(1+(vhat[i,:].dot(inv(matrixR)).dot(np.transpose(vhat[i,:]))))
        weight[i]= 1/(1+(vhat[i,:].dot(np.transpose(vhat[i,:]))))
    weight=weight/sum(weight)
    W.append(weight)
    tim=tim+1
    h=0
    for i in range(mem):
        h=h+weight[i]**2  
    neff=1/h
    NEFF.append(neff)
    cum=np.cumsum(weight) 
#    if neff<mem:
#       from van import particle
#       for i in range(mem):
#        x0_en[:,i]=x_f[particle(cum,mem),:]
#        print(particle(cum,mem))      
#    else:
#        x0_en=x_f
#        x0_en=np.transpose(x0_en)
    if neff<mem/2:
       for k in range(mem):
           j=0
           y=(1/mem)*np.random.uniform(0,1,1)
           for i in range(mem):
               i=i+1
               u=y+(1/mem)*(i-1)
               while (u>cum[j]):
                     j=j+1
               x0_en[:,k]=x_f[j,:]     
    else:
        x0_en=x_f
        x0_en=np.transpose(x0_en)
        
    pos=0
    for i in range(mem):
        pos=pos+weight[i]*x_f[i,:]   
    posmean.append(pos)

 
fig = plt.figure()
ax=plt.axes()
axes = plt.gca()
axes.set_xlim([0,tot])
plt.plot(timestep,truth[:,1])  
for i in range(len(time)):
    plt.plot(time[i],posmean[i][1],'-o',color='blue') 
    plt.plot(time[i],obs[i,1],'-o',color='yellow') 
for i in range(len(time)):
    plt.plot(ts[i][:],pmean[i][:,1],color='black')   

    
for i in range(len(time)):    
    for k in range(mem):        
        if i==len(time)-1:
           plt.plot(ts[i][:],sss[i][k][:,1],color='grey')
        else:
            plt.plot(ts[i][:],sss[i][k][0:-1,1],color='grey')
#
#
#a=np.random.randn(len(timestep),size) 
#for i in range(len(timestep)):
#    a[i,:]=posmean[i][0:size]

#fig, ax = plt.subplots()
#show_patterns(np.transpose(a), ax=ax)
#j=0
#y=(1/mem)*np.random.uniform(0,1,1)
#for i in range(mem):
#    i=i+1
#    u=y+(1/mem)*(i-1)
#    while (u>cum[j]):
#            j=j+1
#    k=j
#    print(k)
    
