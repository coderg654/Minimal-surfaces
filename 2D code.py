# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 04:28:41 2024

@author: gidob
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fft as fft
import copy
import scipy.optimize as opt
import scipy.sparse as sp
import time

####################################
####################################

####
#DEFINING FUNCTIONS
####

#Function that calculates the matrices which correspond to taking the gradient of a scalar field
def matrices():
    ID = np.identity(n)
    off = np.ones(n-1)
    #Roll = np.diag(off,1)-np.diag(off,-1)+np.diag([1],-n+1)-np.diag([1],n-1)
    Roll = np.diag(np.ones(n))-np.diag(off,-1)-np.diag([1],n-1)

    matrix1 = np.zeros([n**2,n**2])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    matrix1[i*n+k,j*n+l]= ID[i,j]*Roll[k,l]
                    
    matrix2 = np.zeros([n**2,n**2])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    matrix2[i*n+k,j*n+l]= Roll[i,j]*ID[k,l]
    #matrix1,matrix2 = matrix1/2,matrix2/2
    return matrix1,matrix2

def altmatrices():
    idd = sp.diags(np.ones(n**2))
    riup = sp.diags(np.ones(n**2-n),n)
    ridown = sp.diags(np.ones(n),-n**2+n)

    ri = -idd+riup+ridown

    up = np.append(np.ones(n-1),np.zeros(1))
    up = np.tile(up,n)[0:n**2-1]
    down = np.append(np.ones(1),np.zeros(n-1))
    down = np.tile(down,n)[0:n**2-n+1]

    irup = sp.diags(up,1)
    irdown = sp.diags(down,1-n)
    ir = -idd+irup+irdown

    ri,ir = ri.tocsc(),ir.tocsc()
    rit,irt = -ri.T,-ir.T
    return ri,ir,rit,irt


#calculate the W matrix used in solving the poisson equation
def make_w(length):
    kx,ky =0,0
    wyz = np.ones([length,length])
    for ix in range(length):
        for iy in range(length):
            for iz in range(length):
                if [ix,iy]!=[0,0]:
                    kx = math.pi*ix/length
                    ky = math.pi*iy/length
                    wyz[ix,iy] = 4*(math.sin(kx)**2+math.sin(ky)**2) # 4 of 2 als factor?
    return wyz 


#solving the poisson equation in the fourier domain
def pois_solv(pois1):
    ###
    #fft poisson solve
    ###
    phix = fft.fftn(pois1)
    phi2 = (1/w) * phix
                    
    ret_fun = fft.ifftn(phi2)
    return ret_fun

#how to plot a vector plot
def vec_plot(a,b,title):
    X = np.arange(0, n, 1)
    Y = np.arange(0, n, 1)
    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, b,a)
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                 label='Quiver key, length = 10', labelpos='E')
    plt.title(title)
    plt.show()

#MAIN PROGRAM
#####

n = 18 # size of the grid


sig_int = np.zeros([n,n]) #initializing the set of boundary points

RI,IR, RIT, IRT = altmatrices() #making the gradient matrices

m1,m2 = 0.5*(RI+RIT), 0.5*(IR+IRT)

w = make_w(n) #making the w matrix used in the poisson solver

###### ADDING BOUNDARIES

#sig_int[20,20] = 1
sig_int[5,5] = 1
sig_int[10,13] = -1
#sig_int[10,20] = -1

Lx = 5
Ly = 8

LY,LX = Lx,-Ly

psi = -np.real(pois_solv(sig_int))
psiflat = psi.flatten()
eta1 = -m2.dot(psiflat)
eta2 = m1.dot(psiflat)

pos = m1.dot(m1.dot(psiflat))+ m2.dot(m2.dot(psiflat))

Lxin = np.sum(eta1)
Lyin = np.sum(eta2)
difx = (LX-Lxin)/(n*n)
dify = (LY-Lyin)/(n*n)
eta1 += difx  #changed around because we are working with rotation, not leg
eta2 += dify

####
#minimization
######

def fun(x):
    h1 = m1.dot(x)+eta1
    h2 = m2.dot(x)+eta2
    g =  (h1**2 + h2**2)
    g2 = g**(0.5)
    g3 = g2**(-1)
    
    H1 = m1.multiply(h1)
    H2 = m2.multiply(h2)
    
    fp = np.sum((H1+H2).multiply(g3),axis=1)
    f = np.sum(g2)
    
    return f

#fun = lambda x: np.sum(((m1.dot(x)+eta1)**2+(m2.dot(x)+eta2)**2)**0.5) ####

meth = 'L-BFGS-B'
t1=time.time()
res = opt.minimize(fun, tuple(np.zeros(n**2)),args=(), method=meth, jac=None,
                   hess=None, hessp=None, bounds=None, constraints=(), tol=None,
                   callback=None, options={'maxfun': 10000000, 'maxiter': 4000, 'disp': False})
# try conjugate gradient or LBFGS
# options={'maxfun': 10000000, 'maxiter': 2000, 'disp': False}      #LBFGSB         41.3536
# options={'maxiter': 1600, 'disp': False}                          #CG   nsuc6     42.0914
# options={'maxiter': 1600, 'disp': False}                          #BFGS ns        41.344
# if search for gradient fprime approx
t2=time.time()
print("time elapsed", round(t2-t1), " seconds. method = ",meth)
print(res)

sol = res['x']
dphi1 = m1.dot(sol)+eta1
dphi2 = m2.dot(sol)+eta2

etaf0 = dphi1.reshape(n, n)
etaf1 = dphi2.reshape(n, n)
Size = (etaf0**2+etaf1**2)**0.5
Sz = np.where(Size-np.min(Size)>0.55*(np.max(Size)-np.min(Size)),Size,0)

######
#recovery of the line:
######
    
dxx = RIT.dot(dphi1)
dyy = IRT.dot(dphi2)
line = np.real(pois_solv(-(dxx+dyy).reshape(n,n)))
RANGE = np.max(line)-np.min(line)
MIN = np.min(line)
lin =np.where(line<0.45*RANGE+MIN,RANGE+MIN,line)
plt.imshow(Size*np.where(0.55*RANGE+MIN<=lin,0,1))
plt.title('line')
plt.show()

#####
#Color plots
#####

#half = int((n-1)/2)
plt.imshow(np.real(pos.reshape(n,n)))
plt.title('signed intersection')
plt.show()

plt.imshow(np.real(sig_int))
plt.title('signed intersection')
plt.show()

plt.imshow(np.real(psi))
plt.title('psi')
plt.show()

########
#Vector Plot
########
# eta1 = eta1.reshape(n,n)
# eta2 = eta2.reshape(n,n)

# vec_plot(eta1,eta2,"initial")

# vec_plot(etaf0,etaf1,"final "+meth)