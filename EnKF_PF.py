import pylab as plt
from numpy import *
import numpy.matlib
import math
from numpy.random import randn
import numpy as np
import scipy.integrate as integrate
from numpy.linalg import inv

tc=100
mu=3
true_mu=3.5  #true value%
X0 =[0.1, 0.2, true_mu]; # initial state for truth
x0= [0.1, 0.2, mu]; #initial state for model
tspan=np.arange(0,tc+0.001,0.01) #timestep
mem=50 # Ensemble Members
sigma=0.1 #Variance


from van import Vanderpolt 
from van import Vanderpol2# THIS IS THE COMMAND, FILE NAME IS van and Funcyion name os VanderPOl2
from scipy.integrate import odeint
xr=odeint(Vanderpolt,X0,tspan)

s1=np.random.normal(0, sigma, mem)
s2=np.random.normal(0, sigma, mem)
s3=np.random.normal(0, 1, mem)
x0_en=np.zeros((mem,3))
x0_en[:,0]=x0[0]+s1
#x0_en[:,0]=np.matlib.repmat(x0[0],mem,1)
x0_en[:,1]=x0[1]+s2
#x0_en[:,1]=np.matlib.repmat(x0[1],mem,1)
x0_en[:,2]=x0[2]+s3
np.savetxt('initial.dat', x0_en)
fixed=x0_en
dt=0.5
timestep=np.arange(dt,tc+0.001,dt)
length=len(timestep)
u=np.zeros((length,2))
R=0.01
matrixR=[[R,0],[0,R]]
H=[[1,0,0],[0,1,0]]
H=np.array(H)
up=[]
t1=50
for k in range(length):  
    up1=np.zeros((mem,2))
    u[k,0]=xr[t1,0]+np.random.normal(0, math.sqrt(R), 1)
    u[k,1]=xr[t1,1]+np.random.normal(0, math.sqrt(R), 1)
    t1=t1+50
    up1[:,0]=u[k,0]+np.random.normal(0, math.sqrt(R), mem)
    up1[:,1]=u[k,1]+np.random.normal(0, math.sqrt(R), mem)
    up.append(up1)
np.savetxt('obs.dat',u)
pert=np.zeros((mem*length,2))

for i in range(length):
    s=i*mem
    k=mem*(i+1)
    pert[s:k,:]=up[i][:]
   
# particle filter
sss=[]
ts=[]
pxf=[]
pmean=[]
posmean=[]
tim=0
totalt=[]
for t in timestep: 
    x_f= np.zeros((mem, 3))  
    s=0;
    ensem=[];
    for i in np.arange(0,mem):
        tspan2=np.arange(t-dt,t+0.001,0.01)
        totalt.append(tspan2)
        XX=x0_en[i]
        a=len(tspan2) 
        z=odeint(Vanderpol2,XX,tspan2)
        s = s+z
        x_f[i,:]= z[-1,:] 
        ensem.append(z)     
    sss.append(ensem)
    ts.append(tspan2)
    pxf.append(x_f)
    mean=s/mem    
    pmean.append(mean)   
    covariance=np.cov(np.transpose(x_f))
    vhat=np.zeros((mem,2))
    vhat=u[tim,:]-x_f[:,0:2]
    tim=tim+1
    weight=np.zeros((mem,1))
    for i in range(mem):
        weight[i]= 1/(1+ vhat[i,:].dot(inv(matrixR)).dot(np.transpose(vhat[i,:])))   
    weight=weight/sum(weight)
    h=0
    for i in range(mem):
        h=h+weight[i]*(weight[i])
    
    neff=1/h
    cum=np.cumsum(weight)
    
    if neff<mem/2:
       from van import particle
       for i in range(mem):
        x0_en[i,:]=x_f[particle(cum,mem),:]
    else:
        x0_en=x_f
    
    if t>=80:
        x0_en=x_f
  
    pos=0
    for i in range(mem):
        pos=pos+weight[i]*x_f[i,:]
    
    posmean.append(pos)


# enkf
ssse=[]
exf=[]
emean=[]
enPmean=[]
tim=0
x0_en=fixed
for t in timestep: 
    enx_f= np.zeros((mem, 3))  
    s=0;
    enseEn=[]
    for i in np.arange(0,mem):
        tspan2=np.arange(t-dt,t+0.001,0.01)
        XX=x0_en[i]
        a=len(tspan2) 
        z=odeint(Vanderpol2,XX,tspan2)
        s = s+z
        enx_f[i,:]= z[-1,:]
        enseEn.append(z)            
    ssse.append(enseEn)
    exf.append(enx_f)
    mean=s/mem    
    emean.append(mean)   
    covariance=np.cov(np.transpose(enx_f))
    vhat=up[tim]
    tim=tim+1
    hdash=np.transpose(H)
    kg=covariance.dot(hdash).dot(inv(H.dot(covariance).dot(hdash)+matrixR))
    ks=kg[0:2,0:2]
    kp=kg[2,:]
    ty=H.dot(np.transpose(enx_f))
    innovation= np.transpose(vhat)-ty
    xa=np.transpose(enx_f)+kg.dot(innovation)
    xa=np.transpose(xa)
    x0_en=xa
    if t>=80:
        x0_en=enx_f
    xamean=np.mean(xa,axis=0)
    enPmean.append(xamean)
    
# figures  
fig = plt.figure()
ax=plt.axes()
plt.plot(tspan,xr[:,0])  
for i in range(len(timestep)):
    plt.plot(timestep[i],posmean[i][:][0],'-o',color='blue') 
for i in range(len(timestep)):
    plt.plot(timestep[i],enPmean[i][:][0],'-o',color='black') 
    plt.plot(timestep[i],u[i,0],'-o',color='yellow') 

plt.legend()
plt.show()

fig = plt.figure()
ax=plt.axes()
for i in range(len(timestep)):
    for k in range(mem):
        plt.plot(ts[i][:],ssse[i][k][:,0],color='grey')
    plt.plot(timestep[i],enPmean[i][:][0],'-o',color='black') 
    plt.plot(timestep[i],u[i,0],'-o',color='yellow') 


fig = plt.figure()
ax=plt.axes()
for i in range(len(timestep)):
    for k in range(mem):
        plt.plot(ts[i][:],ssse[i][k][:,0],color='grey')
    plt.plot(ts[i][:],emean[i][:,0],'-o',color='black') 
    plt.plot(timestep[i],u[i,0],'-o',color='yellow') 

#np.savetxt('output.dat', x0_en)
#JK=np.loadtxt('output.dat')
# 
 
 