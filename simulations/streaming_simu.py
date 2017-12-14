# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:03:06 2015

@author: claire
"""


from my_optimisation import objective,gradient

import scipy as sp
import numpy as np
from scipy import optimize
#from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt

    
def RMSE(theta_hat,theta_star):
    return sp.sqrt(sum((theta_hat-theta_star)**2)/len(theta_star))

def simul_stream(theta,sigma,N):
    theta_hat=np.zeros((K,N))
    theta_ls=np.zeros((K,N))
    data=[[np.random.normal(loc=theta[k],scale=sigma)] for k in range(K)]
   
    beta_exp=np.exp(u1*theta+u0)
    proba_sel=beta_exp/sum(beta_exp)
    cum_prob=np.cumsum(proba_sel)
    
    samples=np.zeros((K,N))
    for i in range(K):
        samples[i,:]=np.random.normal(loc=theta[i],scale=sigma,size=N)
        
 #   for t in range(K):
 #       u=np.random.uniform()
 #       item_sel=sum(cum_prob<u)
 #       data[item_sel].append(samples[i,t])
 #       theta_hat[:,t]=0.5*np.ones(3)
 #       theta_ls[:,t]=0.5*np.ones(3)
        
    for t in range(N):
        u=np.random.uniform()
        item_sel=sum(cum_prob<u)
        data[item_sel].append(samples[i,t])
        nb=np.asarray([np.size(data[i]) for i in range(K)])
        deltas=np.asarray([sum(data[i]) for i in range(K)])
        sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u1,u0,r,nb,deltas,sigma))
        theta_hat[:,t]=sol[0][0:K]
        theta_ls[:,t]=deltas/nb
        
    return (data,theta_hat, theta_ls)
    
    
K=3
init=np.zeros(2*K)    
r=6 #reasonable value
u1=1
u0=1.8
alpha=1

#theta=np.random.uniform(low=0.,high=5.,size=K)
sigma=2.5
N=100
theta=np.array([2,3,4])

## simulate online estimation
NbExp=50
err_hat=np.zeros((NbExp,N))
err_ls=np.zeros((NbExp,N))

for exp in range(NbExp):
    data, theta_hat, theta_ls=simul_stream(theta,sigma,N)
    
    err_hat[exp,:]=np.asarray([RMSE(theta_hat[:,k],theta) for k in range(N)])
    err_ls[exp,:]=np.asarray([RMSE(theta_ls[:,k],theta) for k in range(N)])
    
err_mean_hat=np.mean(err_hat,0)
err_mean_ls=np.mean(err_ls,0)

plt.figure(1)
plt.plot(range(N),err_mean_hat,color='g',label='SB estimator')
plt.plot(range(N),err_mean_ls,color='y',label='LS estimator')
plt.ylabel("RMSE")
plt.xlabel("time")
plt.legend(loc=4)
plt.show()
plt.savefig('RMSE_vs_time.eps', format='eps', dpi=1000)