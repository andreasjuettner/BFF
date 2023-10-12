###############################################################################
# Copyright (C) 2023
#
# Author: Andreas Jüttner andreas.juttner@cern.ch
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# See the full license in the file "LICENSE" in the top level distribution
# directory

# The code has been used for the analysis presented in
# "Nonperturbative infrared finiteness in super-renormalisable scalar quantum
# field theory" https://arxiv.org/abs/2009.14768
##############################################################################
import numpy as np

class zfit_ff():
	'''
	 This class contains basic definitions for BGL and BCL parameterisations
	'''
	def __init__(self,input_dict,large_t=False,large_t_ibase=0):
	 '''
	   input_dict	specifies zfit details in terms of dict
	  
	 '''
	 for k,v in input_dict.items():
	  setattr(self,k,v) 
	 self.tp	= (self.Mi+self.Mo)**2
	 self.tm	= (self.Mi-self.Mo)**2
	 self.tstar	= eval(input_dict['tstar'])
	 self.t0      	= eval(input_dict['t0'])
	
	 self.data 	= []
	 self.large_t	= large_t	# constrain large-t systematics?
	 self.large_t_ibase = large_t_ibase

        ################################################################
	def rho(self,k,j,wo):
	 '''
	  constraint for the large-t behaviour, such that f\lesssim t^-1
	 '''
	 if wo==0:
	  return -0.5*((2+3*j+j**2) - (3+2*j)*k + k**2)
	 elif wo==1:
   	  return      ((  2*j+j**2) - (2+2*j)*k + k**2)
	 elif wo==2:
	  return -0.5*((    j+j**2) - (1+2*j)*k + k**2)
	 else:
	  print("zfit.rho: value j=%d not defined"%j)
	  exit()

	def zfit_BGL_p_pole(self,qsql):
	 B	= 1.
	 for pole in self.mpolep:
	  B	*= self.zfn(qsql,self.tstar,pole**2)
	 phi 	= self.outer_phi_p(qsql)
	 prefac = 1./B/phi
	 return prefac,phi,B

	def zfit_BGL_p_fn(self,alpha,qsq,f0_constraint=False):
	 if f0_constraint:
	  qsql 	= qsq*0
	 else:
	  qsql 	= qsq
	 prefac,phi,B = self.zfit_BGL_p_pole(qsql)
	 z=self.zfn(qsql,self.tstar,self.t0)

	 if f0_constraint: # this is the f(0) kinematic constraint
	  mpole0=self.mpole0
	  chi0  =self.chi0
	  B0=1.
	  B00=1.
	  for pole in mpole0:
	   B00	*= self.zfn(qsq*0,self.tstar,pole**2)
	   B0	*= self.zfn(qsq,self.tstar,pole**2)
	  phi0 	 = self.outer_phi_0(qsq)
	  phi00  = self.outer_phi_0(qsq*0)
	  prefac = B00*phi00/(B0*phi0)/(B*phi)

	 if self.large_t:
	  base_i  = self.large_t_ibase
	  RHO     = lambda k: z**k \
			+ self.rho(k,base_i,0)*z**(base_i+0) \
			+ self.rho(k,base_i,1)*z**(base_i+1) \
			+ self.rho(k,base_i,2)*z**(base_i+2)
	  indices = range(self.Kp+3)
	  indices = np.delete(indices,base_i + np.array([0,1,2]))
	  aind 	  = lambda k: k if k<base_i else k-3 if k>=base_i+3 else 0
	 else:
	  indices = range(self.Kp)
	  RHO     = lambda k: z**k
	  aind    = lambda k: k 
         
	 if alpha==[]:
 	  return (prefac*np.array([RHO(k) for k in indices])).T
	 else:
	  if len(alpha.shape)==2:
	   return (np.sum(prefac*np.array([alpha[:,aind(k)]*RHO(k) for k in indices]),axis=0)).T
	  else:
	   return (np.sum(prefac*np.array([alpha[aind(k)]*RHO(k) for k in indices]),axis=0))
	
        ################################################################
	def zfit_BGL_0_pole(self,qsql):
	 B	= 1.
	 for pole in self.mpole0:
	  B	*= self.zfn(qsql,self.tstar,pole**2)
	 phi 	= self.outer_phi_0(qsql)
	 prefac = 1./B/phi
	 return prefac,phi,B
	def zfit_BGL_0_fn(self,alpha,qsq,f0_constraint=False):
         # construct Blaschke and Pole 
	 prefac,phi,B = self.zfit_BGL_0_pole(qsq)
	 z=self.zfn(qsq,self.tstar,self.t0)
	 
         # the following implements the off-diagonal block of the design matrix 
	 # that implements the f(0) constraint
	 if f0_constraint:
	  phi0 	= self.outer_phi_0(qsq*0)
	  z0	= self.zfn(qsq*0,self.tstar,self.t0)
	  r 	= np.arange(1,self.K0)
	  B0	= 1.
	  for pole in self.mpole0:
	   B0*= self.zfn(qsq*0.,self.tstar,pole**2)
	  prefac0 = 1./B0/phi0
	  res = (prefac*(np.array([z**k for k in r])-np.array([z0**k for k in r]))).T
	 else:
	  if alpha==[]:
	   res = (prefac*np.array([z**k for k in range(self.Kp)])).T
	  else:
	   if len(alpha.shape)==2:
	    res = (np.sum(prefac*np.array([alpha[:,k]*z**k for k in range(self.K0)]),axis=0)).T
	   else:
	    res = (np.sum(prefac*np.array([alpha[k]*z**k for k in range(self.K0)]),axis=0))
	 return res
	
        ################################################################
	def zfit_BCL_p_fn(self,alpha,qsq,special=[]):
	 z	= self.zfn(qsq,self.tstar,self.t0)
	 poles 	= 1.
	 if len(self.mpolep)>0:
	  for pole in self.mpolep:
	   poles*=1./(1-qsq/pole**2)
	 if alpha == []:
	  if special==[]:
	   return poles*np.sum(np.array([(z**k-1.*(-1)**(k-self.Kp)*k/self.Kp*z**self.Kp) for k in range(self.Kp)]),0)
	  elif special=='monomials':
	   return poles*np.array([(z**k-1.*(-1)**(k-self.Kp)*k/self.Kp*z**self.Kp) for k in range(self.Kp)])
	 else:
	  return poles*np.sum(np.array([alpha[k]*(z**k-1.*(-1)**(k-self.Kp)*k/self.Kp*z**self.Kp) for k in range(self.Kp)]),0)

        ################################################################
	def zfit_BCL_0_fn(self,alpha,qsq,special=[]):
	 z	= self.zfn(qsq,self.tstar,self.t0)
	 poles 	= 1.
	 if (len(self.mpole0)>0) and (self.mpole0[0]!=0):
	  for pole in self.mpole0:
	   poles*=1./(1-qsq/pole**2)
	 if alpha == []:
	  if special==[]:
	   res = poles*np.sum(np.array([z**k for k in range(self.K0)]),0)
	  elif special=='monomials':
	   res = poles*np.array([z**k for k in range(self.K0)])
	 else:
	  res = poles*np.sum(np.array([alpha[k]*z**k for k in range(self.K0)]),0)
	 return res
		
        ################################################################
	def outer_phi_p(self,qsq):
	 if self.outer_p[0]!=0:
	  nI 	= self.outer_p[0]
	  K	= eval(self.outer_p[1])
	  a	= self.outer_p[2]
	  b	= self.outer_p[3]
	  return self.outer_phi_ker(qsq,a,b,nI,K,self.chip)
	 else:
	  return 1.  
	
        ################################################################
	def outer_phi_0(self,qsq):
	 if self.outer_0[0]!=0:
	  nI 	= self.outer_0[0]
	  K	= eval(self.outer_0[1])
	  a	= self.outer_0[2]
	  b	= self.outer_0[3]
	  return self.outer_phi_ker(qsq,a,b,nI,K,self.chi0)
	 else:
	  return 1.
	
        ################################################################
	def outer_phi_ker(self,qsq,a,b,nI,K,chi):
	 rq	= np.sqrt(self.tstar - qsq)
	 r0	= np.sqrt(self.tstar - self.t0)
	 rm	= np.sqrt(self.tstar - self.tm)
	 res = np.sqrt(nI/K/chi)\
		* np.sqrt(rq)/np.sqrt(r0)\
		* (rq 		+ r0			)\
		* (rq 		+ np.sqrt(self.tstar)	)**(-(b+3))\
		* (self.tp 	- qsq			)**(a*1./4)\
		* (rq 		+ rm			)**(a*1./2)
	 return res

        ################################################################
	def Verblunsky_coefficients(self,order,argz):
	 # The Verblunsky coefficients are a property of the measure on the unit circle. A 
	 # simple recursion relation can be derived by orthogonalising via Gram-Schmidt 
	 # basis function z=exp(i n alpha), for which the projection 
	 # integrals can be solved analytically:
	 # \int_{-argz}^{+argz} exp(-i n alpha)  = 2 sin(argz n) / n
	 # Note that the r.h.s. generally real
	 # The resulting Verblunsky coefficients are then valid for any more general 
	 # polynomials in z

	 # Norm on part of unit cirlce (-argz,+argz)
	 N	= 2 * argz
	
         # Initialise recursion
	 Pp		= np.zeros((order+1,order+1))
	 Pm		= np.zeros((order+1,order+1))
	 Np		= np.zeros((order,))
	 Verblunsky	= np.zeros((order,))
	 Pp[0,0]	= N
	 Pm[0,0]	= N
	
	 # using orthogonality
	 Pp[0,1:]	= [2. / n * np.sin(0.5 * N * n) for n in range(1,order+1)]
	 Pm[0,1:]	= Pp[0,1:]
	 	
	 Verblunsky[0]	= Pp[0,1] / Pm[0,0]	
	 Np[0]		= np.sqrt(2 * argz)
	 for n in range(1,order):
	  for i in range(order):
	   Pp[n,i]	= Pp[n-1,i+1]	- Verblunsky[n-1] * Pm[n-1,i]	
	   Pm[n,i]	= Pm[n-1,i  ]	- Verblunsky[n-1] * Pp[n-1,i+1]	
	  Verblunsky[n] = Pp[n][1] / Pm[n,0]
	  Np[n]		= np.sqrt(2 * argz * np.prod([1.-Verblunsky[i]**2 for i in range(n)]))
	 #print('Pp',Pp[:order,:order])
	 #print('Pm',Pm[:order,:order])
	 print("Verblunsky coefficients ",Verblunsky)
	 print("normalisation           ",Np)
	 self.Verblunsky_rho = Verblunsky
	 self.Verblunsky_N   = Np

	def Szegoe_match(self,order,argz):
	 # function returns 
	 # B 	matrix such that B.z gives polynomials orthogonal on arc [-arcz,+arcz]
	 # M	matrix such that aT.M.a gives correct unitarity constraint, where a are 
	 # 	coefficients of basis {1,z,z^2,z^3,...} rather than Szegoe polynomials.
	 import sympy as sy
	 Pp	= np.zeros((order+1,order+1))
	 Pm	= np.zeros((order+1,order+1))
	 rho	= self.Verblunsky_rho
	 Pp[0]	= 2 * argz
	 Pm[0]	= 2 * argz
	 z	= sy.symbols('z')
	 B	= np.identity(order)
	 Pp	= 1
	 Pm	= 1
	 Np	= self.Verblunsky_N
	 B[0,0]	= 1./Np[0]
	 for n in range(1,order):
	   Pp0	=   z * Pp - rho[n-1] *     Pm	
	   Pm0	=       Pm - rho[n-1] * z * Pp	
	   Pp	= Pp0
	   Pm	= Pm0
	   B[n,:n]	= sy.Poly(Pp/Np[n]).all_coeffs()[n:0:-1]
	   B[n,n]	= 1./Np[n]
	 invB	= np.linalg.inv(B)
	 M	= np.dot(invB,invB.T) 
	 return B,M

        ################################################################
	def zfn(self,qsq,tp,t0,epsilon=0):
	 rt	= np.sqrt(tp - qsq + epsilon) # added zero imaginary to allow sqrt of negativ number
	 r0	= np.sqrt(tp - t0  + epsilon)
	 return (rt - r0) / (rt + r0)

        ################################################################
	def make_b0_BGL(self,alpha):
	 # implement contsraint f0(0)=f+(0)
	 if len(alpha.shape)==2:
 	  fp = self.zfit_BGL_p_fn(alpha[:,:self.Kp],0)
	 else:
 	  fp = self.zfit_BGL_p_fn(alpha[:self.Kp],0)
	 # compute f0 with out b[0]
	 B0=1.

	 for pole in self.mpole0:
	  B0*= self.zfn(0,self.tstar,pole**2)
	
	 phi0    = self.outer_phi_0(0)
	 z       = self.zfn(0,self.tstar,self.t0)
	 if len(alpha.shape)==2:
	  f0_bar0 = np.sum([alpha[:,self.Kp+k]*z**(k+1) for k in range(0,self.K0-1)],0)
	 else:
	  f0_bar0 = np.sum([alpha[self.Kp+k]*z**(k+1) for k in range(0,self.K0-1)],0)
	 a       = fp*B0*phi0-f0_bar0
	 return a
	
	def Cff_BCL(self,qsqp,qsq0,covp0,bp,b0):
	  # compute covariance cov(f(E,E')) from HMChPT parameters
	  N     = len(qsqp)+len(qsq0)
	  Kp	= self.Kp
	  K0	= self.K0
	  Np	= len(qsqp)
	  N0	= len(qsq0)
	  Cf    = np.zeros((N,N))
	  for i in range(Np):
	   for j in range(Np):
	    Gvec0 = self.zfit_BCL_p_fn([],qsqp[i],special='monomials')
	    Gvec1 = self.zfit_BCL_p_fn([],qsqp[j],special='monomials')
	    Cf[i,j] = np.dot(Gvec0,np.dot(covp0[:Kp,:Kp],Gvec1))
	  for i in range(N0):
	   for j in range(N0):
	    Gvec0 = self.zfit_BCL_0_fn([],qsq0[i],special='monomials')
	    Gvec1 = self.zfit_BCL_0_fn([],qsq0[j],special='monomials')
	    Cf[Np+i,Np+j] = np.dot(Gvec0,np.dot(covp0[Kp:,Kp:],Gvec1))
	  for i in range(Np):
	   for j in range(N0):
	    Gvec0 = self.zfit_BCL_p_fn([],qsqp[i],special='monomials')
	    Gvec1 = self.zfit_BCL_0_fn([],qsq0[j],special='monomials')
	    dum   = np.dot(Gvec0,np.dot(covp0[:Kp,Kp:],Gvec1))
	    Cf[i,Np+j] = dum
	    Cf[Np+j,i] = dum
	  return Cf

	def E_qsq(self,qsq):
	   """
	    return final-state-energy for given qsq
	   """
	   ksq        = (self.Mi**4+(self.Mo**2-qsq)**2-2*self.Mi**2*(self.Mo**2+qsq))/(4*self.Mi**2)
	   return  np.sqrt(self.Mo**2+ksq)

	def G_qsq(self,qsq,pole,coeff):
	  # construct ff from HMChPT in continuum limit
	  Evec = self.E_qsq(qsq)
	  return np.array([1./(Evec+pole)*Evec**i for i in range(len(coeff))])

	def ksq_qsq(self,qsq):
	  return (self.Mi**4+(self.Mo**2-qsq)**2-2*self.Mi**2*(self.Mo**2+qsq))/(4*self.Mi**2) 

	def ff_HMChPT(self,qsq,pole,coeff):
	  # construct ff from HMChPT in continuum limit
	  ksq  = (self.Mi**4+(self.Mo**2-qsq)**2-2*self.Mi**2*(self.Mo**2+qsq))/(4*self.Mi**2)
	  Evec = np.sqrt(self.Mo**2+ksq)
	  return [1./(E+pole)*np.sum([E**i*coeff[i] for i in range(len(coeff))]) for E in Evec]

	def Cff_HMChPT(self,qsqp,qsq0,covp0,cp_BstoK,c0_BstoK):
	  # compute covariance cov(f(E,E')) from HMChPT parameters
	  Nqsqp	= len(qsqp)
	  Nqsq0	= len(qsq0)
	  N     = len(qsqp)+len(qsq0)
	  Np    = len(cp_BstoK)
	  N0    = len(c0_BstoK)
	  Cf    = np.zeros((N,N))
	  import physical_constants as pc
	  for i in range(Nqsqp):
	   for j in range(Nqsqp):
	    Gvec0 = self.G_qsq(qsqp[i],pc.Deltaperp,cp_BstoK[:Np])
	    Gvec1 = self.G_qsq(qsqp[j],pc.Deltaperp ,cp_BstoK[:Np])
	    Cf[i,j] = np.dot(Gvec0,np.dot(covp0[:Np,:Np],Gvec1))
	  for i in range(Nqsq0):
	   for j in range(Nqsq0):
	    Gvec0 = self.G_qsq(qsq0[i],pc.Deltapar,c0_BstoK[:N0])
	    Gvec1 = self.G_qsq(qsq0[j],pc.Deltapar,c0_BstoK[:N0])
	    Cf[Nqsqp+i,Nqsqp+j] = np.dot(Gvec0,np.dot(covp0[Np:,Np:],Gvec1))
	  for i in range(Nqsqp):
	   for j in range(Nqsq0):
	    Gvec0 = self.G_qsq(qsqp[i],pc.Deltaperp,cp_BstoK[:Np])
	    Gvec1 = self.G_qsq(qsq0[j],pc.Deltapar,c0_BstoK[:N0])
	    dum   = np.dot(Gvec0,np.dot(covp0[:Np,Np:],Gvec1))
	    Cf[i,Nqsqp+j] = dum
	    Cf[Nqsqp+j,i] = dum
	  return Cf


