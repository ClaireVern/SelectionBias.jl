# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:04:19 2015

@author: vernade
"""


from my_optimisation import objective,gradient

import scipy as sp
import numpy as np
from scipy import optimize
#from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt

    
def RMSE(theta_hat,theta_star):
    return sp.sqrt(sum((theta_hat-theta_star)**2)/len(theta_star))

def generate(theta,sigma,N,sigma_b,u1):
    K=np.size(theta)
    n=np.zeros(K)
    #n=np.ones(K)
    #theta0=2.7
    delta=np.zeros(K)
    #delta=sigma*np.random.randn(K)+theta # one sample per category
    beta_star=sigma_b*np.random.randn(K)+theta/float(u1)
    exp_beta_star=np.exp(beta_star)
    proba_sel=exp_beta_star/sum(exp_beta_star)
    cum_prob=np.cumsum(proba_sel)
    
    for i in range(N-K):
        u=np.random.uniform()
        item_sel=sum(cum_prob<u)
        n[item_sel]+=1
        delta[item_sel]+=np.random.normal(loc=theta[item_sel],scale=sigma)
    return (n,delta)

#==============================================================================
# First tests 


# # Generate the data
# u1=20
# K=5
# theta_star=np.array((1,2,3,4,5))
# 
# sigma=1.2
# sigma_b=1 # be careful with the noise 
# 
# N=200
# n,delta=generate(theta_star,sigma,N,sigma_b,u1)
# 
# ## Tune r
# theta_ls=delta/n
# proba_est=n/sum(n)
# proba_star=np.exp((theta_star-2.7)/u1)/sum(np.exp((theta_star-2.7)/u1))
# init=np.hstack((theta_ls,0.8*theta_ls/u1))  
# R=np.linspace(0.2,10,20)
# err=np.zeros(20)
# iter_r=0
# for r in R:
#     sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u1,r,n,delta,sigma))
#     theta_hat=sol[0][0:K]
#     err[iter_r]=RMSE(theta_hat,theta_star)
#     iter_r+=1
#     
# 
# plt.figure(1)
# plt.plot(R,err)
# plt.ylabel("RMSE")
# plt.xlabel("regularisation parameter r")
# 
# 
# r_min=R[np.argmin(err)]
# 
# # Learning
# 
# sol1=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u1,r_min,n,delta,sigma))
# theta_hat=sol1[0][0:K]
# beta_hat=sol1[0][K:]
# 
# 
# # plot results 
# 
# ind = np.arange(K)  # the x locations for the groups
# width = 0.25       # the width of the bars
# 
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, theta_star, width, color='r',label='theta_star') #add yerr=... to display variance
# rects2 = ax.bar(ind+width, theta_hat, width, color='g',label='theta_hat')
# rects3 = ax.bar(ind+2*width, theta_ls, width, color='y',label='theta_ls')
# #ax.legend( (rects1[0], rects2[0],rects3[0]), ('theta_star', 'theta_hat','theta_ls') ,loc=3)
# #plt.xlabel("component the parameter vector")
# plt.ylabel("value of the component")
# 
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# 
# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

       

#plt.show()
#plt.savefig('bar_theta.eps', format='eps', dpi=1000)
#==============================================================================

#==============================================================================
# # general r tuning
#==============================================================================

nb_exp=200
N=2000
theta_star=np.array((3.4,3.9,4.6))
K=3
u1=0.3
sigma=1.5
sigma_b=1

ns=np.zeros((nb_exp,K))
R=np.linspace(0.2,10,20)
errs=np.zeros((nb_exp,20))
thetas_hat=np.zeros((nb_exp,K))
thetas_ls=np.zeros((nb_exp,K))
betas_hat=np.zeros((nb_exp,K))

for exp in range(nb_exp):
    n,delta=generate(theta_star,sigma,N,sigma_b,u1)
    ns[exp,:]=n
    init=np.zeros(2*K)
    init=np.hstack((delta/n,0.8*delta/n/u1))   
    iter_r=0
    for r in R:
        sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u1,r,n,delta,sigma))
        theta_hat=sol[0][0:K]        
        errs[exp,iter_r]=RMSE(theta_hat,theta_star)
        iter_r+=1
    
    r_min=R[np.argmin(errs[exp,:])]
    
    sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(0.4,r_min,n,delta,sigma))
    thetas_hat[exp,:]=sol[0][0:K]
    betas_hat[exp,:]=sol[0][K:]
    thetas_ls[exp,:]=delta/n   

ns_avg=np.mean(ns,0)
mthetas = np.ma.masked_array(thetas_hat,np.isnan(thetas_hat))
theta_hat_avg=np.mean(mthetas,0)
hat_err=np.std(mthetas,0)

#beta_hat_avg=np.mean(betas_hat,0)


mls=np.ma.masked_array(thetas_ls,np.isnan(thetas_ls))
theta_ls_avg=np.mean(mls,0)  
ls_err=np.std(mls,0)

merrs= np.ma.masked_array(errs,np.isnan(errs)) 
err_avg=np.mean(merrs,0)
plt.figure(1)
plt.plot(R,err_avg)
plt.ylabel("RMSE")
plt.xlabel("regularisation parameter r")
plt.savefig('error_vs_r.eps', format='eps', dpi=1000)

#plot results averaged

ind = np.arange(K)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, theta_star, width, color='r',label='theta_star') #add yerr=... to display variance
rects2 = ax.bar(ind+width, theta_hat_avg, width, color='g',label='theta_hat',yerr=hat_err)
rects3 = ax.bar(ind+2*width, theta_ls_avg, width, color='y',label='theta_ls',yerr=ls_err)
#ax.legend( (rects1[0], rects2[0],rects3[0]), ('theta_star', 'theta_hat','theta_ls') ,loc=3)
#plt.xlabel("component the parameter vector")
plt.ylabel("value of the component")

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=5)

plt.show()


#==============================================================================
# # influence of N
#==============================================================================



Ns=range(20,200,10)
nb_exp=50
err_ls=np.zeros((nb_exp,np.size(Ns)))
err_hat=np.zeros((nb_exp,np.size(Ns)))
errs=np.zeros(20)
K=2
theta_star=np.array((3.8,4.5))
R=np.linspace(0.2,10,20)

for exp in range(nb_exp):
    
    iterN=0
    for nb_dat in Ns:
        n,delta=generate(theta_star,sigma,nb_dat,sigma_b,u1)
        init=np.hstack((delta/n,delta/n/u1))   
        iter_r=0
        for r in R:
            sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u1,r,n,delta,sigma))
            theta_hat=sol[0][0:K]
            errs[iter_r]=RMSE(theta_hat,theta_star)
            iter_r+=1
        r_min=R[np.argmin(errs)]
        sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u1,r_min,n,delta,sigma))
        theta_hat=sol[0][0:K]
        theta_ls=delta/n
        #theta_ls[np.isnan(theta_ls)]=2.5 # replace Nan with average values
        err_ls[exp,iterN]=RMSE(theta_ls,theta_star)
        err_hat[exp,iterN]=RMSE(theta_hat,theta_star)
        iterN+=1
    

merr_ls=np.ma.masked_array(err_ls,np.isnan(err_ls))
merr_hat=np.ma.masked_array(err_hat,np.isnan(err_hat))

errM_ls=np.mean(merr_ls,0)
errM_hat=np.mean(merr_hat,0) 
hatN_std=np.std(merr_hat,0)
lsN_std=np.std(merr_ls,0)

plt.figure(4)
plt.errorbar(Ns,errM_hat,yerr=hatN_std,color='g',label='SB estimator',linewidth=2.5)
plt.errorbar(Ns,errM_ls,yerr=lsN_std,color='y',label='LS estimator',linewidth=1.8)
plt.ylabel("RMSE")
plt.xlabel("number of data points generated")
plt.legend()
plt.savefig('RMSE_vs_N.eps', format='eps', dpi=1000)

#==============================================================================
# # influence of u1 
#==============================================================================
K=3
theta_star=np.array((3.8,4.1,4.3))

u1_star=0.35

init=np.zeros(2*K)  
R=np.linspace(0.2,10,10)
sigma_b=0.25
Nb_exp=200
N=1000
U1_range=20
err_u1=np.zeros((Nb_exp,U1_range))

for exp in range(Nb_exp):
    
    n,delta=generate(theta_star,sigma,N,sigma_b,u1_star)
        
    iter_u=0
    for u in np.logspace(-1,np.log10(2),U1_range):
        r_current=0
        sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u,r_current,n,delta,sigma))
        theta_hat=sol[0][0:K]
        err_r_current=RMSE(theta_hat,theta_star)
        #tune r
        for r in R:    
            sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u,r,n,delta,sigma))
            theta_hat=sol[0][0:K]
            oks=~np.isnan(theta_hat)
            err_r=RMSE(theta_hat[oks],theta_star[oks])
            if err_r<err_r_current:
                r_current=r
                err_r_current=err_r
        sol=optimize.fmin_l_bfgs_b(objective, init, fprime=gradient,args=(u,r_current,n,delta,sigma))
        theta_hat=sol[0][0:K]    
        oks=~np.isnan(theta_hat)
        err_u1[exp,iter_u]=RMSE(theta_hat[oks],theta_star[oks])
        iter_u+=1
 

merr_u1=np.ma.masked_array(err_u1,np.isnan(err_u1))

      
err_u1_avg=np.mean(merr_u1,0)
plt.figure(5)
plt.plot(np.logspace(-1,np.log10(2),U1_range),err_u1_avg,label='u1*=0.35')
plt.ylabel("RMSE")
plt.xlabel("value of u1 used for learning")
plt.legend()




























