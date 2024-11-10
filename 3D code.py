# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:00:21 2024

@author: gidob
"""
import numpy as np
np.bool = np.bool_
import pyvista as pv
import matplotlib.pyplot as plt
import math
import scipy.fft as fft
import copy
import scipy.optimize as opt
import scipy.sparse as sp
import time




def boundary(t):
    x = 0.5 + 0.2*(np.cos(2*math.pi*t)**power)
    y = 0.5 + 0.2*(np.sin(2*math.pi*t)**power)
    z = 0.5 + np.cos(4*math.pi*t)*0.2
    return np.array([x,y,z]) 

def signed_int(t):
    bt = boundary(t)
    
    b = bt*n
    b = b.astype(int)
    dif = b[:,1:]-b[:,:-1]
    dx = np.zeros([dim,n,n,n])
    for i in range(len(dif[0])):
        x_index = min(b[0,i],b[0,i+1])
        y_index = min(b[1,i],b[1,i+1])
        z_index = min(b[2,i],b[2,i+1])
        dx[:,x_index,y_index,z_index] += dif[:,i]      
    return dx

def make_w(length):
    kx,ky,kz =0,0,0
    wyz = np.ones([length,length,length])
    for ix in range(length):
        for iy in range(length):
            for iz in range(length):
                if [ix,iy,iz]!=[0,0,0]:
                    kx = math.pi*ix/length
                    ky = math.pi*iy/length
                    kz = math.pi*iz/length
                    wyz[ix,iy,iz] = 4*(math.sin(kx)**2+math.sin(ky)**2+math.sin(kz)**2)
    return wyz    

def pois_solv(pois1):
    ###
    #fft poisson solve
    ###
    phix = fft.fftn(pois1)
    phix = (1/w) *phix
                    
    ret_fun = fft.ifftn(phix)
    return ret_fun

def rot(form):
    xgrad = np.gradient(form[0])
    ygrad = np.gradient(form[1])
    zgrad = np.gradient(form[2])
    curl = np.array([zgrad[1]-ygrad[2],xgrad[2]-zgrad[0],ygrad[0]-xgrad[1]])
    return curl

def altmatrices():
    idd = sp.diags(np.ones(n3))
    riiup = sp.diags(np.ones(n3-n2),n2)
    riidown = sp.diags(np.ones(n2),-n3+n2)

    rii = idd-riiup-riidown

    up = np.append(np.ones(n2-n),np.zeros(n))
    up = np.tile(up,n)[0:n3-n]
    down = np.append(np.ones(n),np.zeros(n2-n))
    down = np.tile(down,n)[0:n3-n2+n]

    iriup = sp.diags(up,n)
    iridown = sp.diags(down,n-n2)
    iri = idd-iriup-iridown

    up = np.append(np.ones(n-1),np.zeros(1))
    up = np.tile(up,n2)[0:n3-1]
    down = np.append(np.ones(1),np.zeros(n-1))
    down = np.tile(down,n2)[0:n3-n+1]

    iirup = sp.diags(up,1)
    iirdown = sp.diags(down,-n+1)

    iir = idd-iirup-iirdown

    rii,iri,iir = rii.tocsc(),iri.tocsc(),iir.tocsc()
    riit,irit,iirt = rii.T,iri.T,iir.T
    return rii,iri,iir,riit,irit,iirt
    

def matrices():
    RII = np.identity(n3)-np.diag(np.ones(n2*(n-1)),n2)-np.diag(np.ones(n2),-(n2*(n-1)))
    RI = np.identity(n2)-np.diag(np.ones(n*(n-1)),n)-np.diag(np.ones(n),-(n*(n-1)))
    R = np.identity(n)-np.diag(np.ones(n-1),1)-np.diag([1],-n+1)


    IR = np.zeros([n,n,n,n])
    for i in range(n):
        IR[i,i] = R
        
    IRI = np.zeros([n,n,n2,n2])
    IIR = np.zeros([n,n,n,n,n,n])
    for i in range(n):
        IRI[i,i] = RI
        IIR[i,i] = IR

    IRI = np.hstack(np.hstack(IRI))
        
    IIR = np.hstack(np.hstack(np.hstack(np.hstack(IIR))))  
    
    rii,iri,iir = sp.csc_matrix(RII), sp.csc_matrix(IRI), sp.csc_matrix(IIR)
    riit,irit,iirt = rii.T,iri.T,iir.T
    return rii,iri,iir,riit,irit,iirt
    
    
#####
#initialize:

time0 = time.time()
    
n = 63
n2 = n**2
n3 = n**3
n4 = n**4

t = np.linspace(0,1,1000)

power = 1

dim = 3

w = make_w(n)

m1,m2,m3,M1,M2,M3 = altmatrices()
#A,B,C = m1,m2,m3
##################
#Main:
##################

print(time.time()-time0," seconds have passed")
print("main")

######
#initial guess
dg = signed_int(t)

#find psi such that laplace(psi)= dg
form = np.shape(dg)
psi = np.zeros(form)
for i in range(dim):
    psi[i] = pois_solv(dg[i])
psi = np.real(psi) 
   
#eta = rot(psi)
eta = np.zeros(form)
    
eta = rot(psi)    
    
#make sure the harmonic fields coïncide
Harm_x=0
Harm_y=0
Harm_z=np.pi*0.2**2

int_x = np.average(eta[0])
int_y = np.average(eta[1])
int_z = np.average(eta[2])

eta[0] += (Harm_x-int_x)
eta[1] += (Harm_y-int_y)
eta[2] += (Harm_z-int_z)

eta1,eta2,eta3 = eta[0].flatten(),eta[1].flatten(),eta[2].flatten()
#a,b,c = eta1,eta2,eta3


######
#minimization
######
print(time.time()-time0," seconds have passed")
print("minimizing")


def fun(x):
    h1 = m1.dot(x)+eta1
    h2 = m2.dot(x)+eta2
    h3 = m3.dot(x)+eta3
    g =  (h1)**2 + (h2)**2 + (h3)**2
    
    g2 = g**(-0.5)
    
    H1 = m1.multiply(h1)
    H2 = m2.multiply(h2)
    H3 = m3.multiply(h3) 
    
    fp = np.sum((H1+H2+H3).multiply(g2),axis=1)
    
    f = np.sum(g**0.5)
    
    return f,fp


#fun = lambda x: np.sum(((m1.dot(x)+eta1)**2+(m2.dot(x)+eta2)**2+ (m3.dot(x)+eta3)**2)**0.5) ####

meth = 'L-BFGS-B'
t1=time.time()
res = opt.minimize(fun, tuple(np.zeros(n3)),args=(), method=meth, jac=True,
                   hess=None, hessp=None, bounds=None, constraints=(), tol=None,
                   callback=None, options={'maxfun': 10000000, 'maxiter': 2000, 'disp': False})
# try conjugate gradient or LBFGS
# options={'maxfun': 10000000, 'maxiter': 2000, 'disp': False}      #LBFGSB         41.3536
# options={'maxiter': 1600, 'disp': False}                          #CG   nsuc6     42.0914
# options={'maxiter': 1600, 'disp': False}                          #BFGS ns        41.344
# if search for gradient fprime approx
t2=time.time()
print("time elapsed", round(t2-t1), " seconds. method = ",meth)
print(res)

#####
#computing results
#####

print(time.time()-time0," seconds have passed")
print("computing certain results")


sol = res['x']
dphi1 = m1.dot(sol)+eta1
dphi2 = m2.dot(sol)+eta2
dphi3 = m3.dot(sol)+eta3

solution1 = dphi1.reshape(n,n,n, -1).T[0]
solution2 = dphi2.reshape(n,n,n, -1).T[0]
solution3 = dphi3.reshape(n,n,n,-1).T[0]

dsigma = np.array([solution1,solution2,solution3])
Size = (dsigma[0]**2+dsigma[1]**2+dsigma[2]**2)**0.5

DDU = -(M1.dot(dphi1)+M2.dot(dphi2)+M3.dot(dphi3))
DDU = DDU.reshape(n,n,n,-1).T[0]
line = np.real(pois_solv(DDU))

level = np.where(line<(0.475*(np.max(line)-np.min(line))+np.min(line)),1,0)

# RANGE = np.max(line)-np.min(line)
# MIN = np.min(line)
# line =np.where(line<0.4*RANGE+MIN,RANGE+MIN,0)
# U = np.where(0.6*RANGE+MIN<=lin,0,1)

x = np.linspace(0, 1, n+1)
y = np.linspace(0, 1, n+1)
z = np.linspace(0, 1, n+1)
xv, yv, zv = np.meshgrid(x,y,z)
XYZ = np.meshgrid(x,y,z)

MAX = np.max(Size)
MIN = np.min(Size)


def plotting():
    alpha = 0.3
    ALPHA = True
    beta = 0.4
    BETA = True
    while alpha >-1:

        level = np.where(line<(beta*(np.max(line)-np.min(line))+np.min(line)),1,0)
        Sz = np.where(Size>(alpha*(MAX-MIN)+MIN),1,0)

        #ind = np.argwhere(Size>(alpha*MAX+MIN))
        ind = np.argwhere(level*Sz == 1)
        ind = ind/n

        points = pv.PolyData(ind)
        points = pv.wrap(points)

        surf = points.reconstruct_surface()


        pl = pv.Plotter(shape=(1, 2))
        _ = pl.add_mesh(points)
        _ = pl.add_title('Point Cloud of 3D Surface')
        pl.subplot(0, 1)
        _ = pl.add_mesh(surf, color=True, show_edges=True)
        _ = pl.add_title('Reconstructed Surface')
        pl.show()
        
        if ALPHA:
            x = input("ALPHA = ? ")
        if x == 'done' or x == 'd':
            ALPHA = False
        elif x == '+':
            alpha+=0.1
            continue
        elif x == '-':
            alpha-=0.1
            continue
        else:
            alpha = float(x)
            
        if BETA:
            y = input("BETA = ? ")
            
        if y == 'done' or x == 'd':
            alpha = -1
        elif y == '+':
            beta+=0.1
            continue
        elif y == '-':
            beta-=0.1
            continue
        else:
            beta = float(y)
        

#plotting()