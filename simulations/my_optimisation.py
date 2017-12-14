# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:21:54 2015

@author: vernade
"""


import numpy as np


# functions

def likelihood(theta,beta,n,delta,sigma):
    
    return sum(theta**2*n/(2*sigma**2)-theta*delta/(sigma**2)-n*(beta))+sum(n)*np.log(sum(np.exp(beta)))
   
def objective(param,u1,r,n,delta,sigma):
    size=param.size
    theta=param[:size/2]
    beta=param[size/2:]
    penal=r*sum((u1*(beta)-theta)**2)
    return likelihood(theta,beta,n,delta,sigma)+penal
    
def gradient(param,u1,r,n,delta,sigma):
    size=param.size
    theta=param[:size/2]
    beta=param[size/2:]
    grad1=theta*n/(sigma**2)-delta/(sigma**2)+2*r*(theta-u1*(beta))
    grad2=-n+sum(n)*np.exp(beta)/sum(np.exp(beta))-2*r*u1*(theta-u1*(beta))
   
    gradtot=np.hstack((grad1,grad2))
    # normgrad=sqrt(np.linalg.norm(gradtot)) 
    return gradtot
    
