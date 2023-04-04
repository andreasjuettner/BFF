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
import lib.zfit_lib as zfit_lib
import lib.physical_constants as pc
import numpy as np
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt
import getdist
from getdist import plots
import os
import git
import particles as pt
import h5py
###############################################################################
# Some plot settings
###############################################################################
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans serif",
    "font.size": 16,
})
plt.rcParams['font.family'] = 'sans serif'
# activate tick marks on both lhs and rhs in plots
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left']  = plt.rcParams['ytick.labelleft'] = True

###############################################################################
# the BFF library
###############################################################################
class Bff():
        ################################################################
	def __init__(self,input_dict):
	 # input_dict contains parameters for target BGL function
	 # assign parameters to class-internal variables
	 for k,v in input_dict.items():
	  setattr(self,k,v) 
	 # keep a copy of the input_dict, just in case
	 self.input_dict= input_dict
	 
	 # basic z-fit parameters
	 self.tp	= (self.Mi+self.Mo)**2
	 self.tm	= (self.Mi-self.Mo)**2
	 self.tstar	= eval(input_dict['tstar'])
	 self.t0      	= eval(input_dict['t0'])
	 # set seed for RNG
	 self.rnd	= np.random
	 self.rnd.seed(self.seed)
	 # data is a list that will hold all data sets that will enter the fit
	 self.data 	= []
	 # instantiate the zfit library, which contains functions specific to BGL and BCL fits
	 self.ff	= zfit_lib.zfit_ff(input_dict) # instantiate ff library
	 # MISC initiatlisations	 
	 self.tags	= []		# tags identifying the input data sets
	 self.labels	= []		# labels for data sets to be used in legends
	 self.bfrequ	= 0		# will later be set to one once frequentist fit done
	 self.frequ_cases = []		# list will contain (Kp,K0) pairs for which frequ. fit was done

	 self.Kcases	= []	 	# list will contain (Kp,K0) cases for which inference was done
	 self.report	= {}		# report will contain all sort of generated data
	 self.triangle	= 1		# set to one if triangle-correlation plots to be generated

	
	 self.legendfontsize = 10
	 
	 # create directories for report if needed
	 if not os.path.exists('Report'):
	  os.system('mkdir Report')
	  os.system('mkdir Report/plots')
	  os.system('mkdir Report/tables')
	  print('Created directory Report/')
	  print('Created directory Report/plots/')
	  print('Created directory Report/tables/')

	 self.separator = '##########################################'
	 self.longseparator = self.separator+self.separator
	 print(self.longseparator)
	 print('Starting BFF')
	 print('Target statistics is %d'%(self.N))
	 print(self.longseparator)

        ################################################################
	def add_data(self,data_dict,tag):
	 '''
	  this function will read in input-data sets in dictionary format contained in 
	  data_dict. Depending on the input-data format it will do 
	  some data manipulation to bring it into a unfiorm format for 
	  further processing. The resulting data is appended to list self.data
	 '''
	 self.tags.append(tag) 
	 self.data_dict	= data_dict
	 dd 		= data_dict[tag]
	 dd['tag']	= tag
	 dd['label']	= data_dict[tag]['label']
	 self.labels.append(data_dict[tag]['label'])
	 qsqp		= dd['qsqp']
	 qsq0		= dd['qsq0']

	 if dd['data type'] == 'HMChPT':
	  # This assumes input in the format defined by equation (27) of RBC/UKQCD 23
	  # the HMChPT functions are defined in zfit_lib.py
	  # The following computes fp and f0 at reference qsq values
	  qsqp_cont	= np.linspace(min(qsqp),max(qsqp),100) # these qsq arrays will be used
	  qsq0_cont	= np.linspace(min(qsq0),max(qsq0),100) # for plotting
	  fp_lambda	= lambda a:ff.ff_HMChPT(qsqp_cont,dd['Deltaperp'],dd['cp'])
	  f0_lambda	= lambda a:ff.ff_HMChPT(qsq0_cont,dd['Deltaperp'],dd['c0'])
	  ff		= zfit_lib.zfit_ff(self.input_dict) # instantiate zfit_lib
	  qsqp		= dd['qsqp']
	  qsq0		= dd['qsq0']
	  dd['fp'] 	= ff.ff_HMChPT(qsqp,dd['Deltaperp'],dd['cp'])		# compute fp central values
	  dd['f0'] 	= ff.ff_HMChPT(qsq0,dd['Deltapar' ],dd['c0'])		# compute f0 central values
	  dd['Cff']	= ff.Cff_HMChPT(qsqp,qsq0,dd['Cp0'],dd['cp'],dd['c0'])  # compute covariance matrix 
	  del ff

	 elif dd['data type'] == 'BCL':
	  # This assumes BCL coefficients and parameters (like e.g. poles) to be provided
	  # and computes fp and f0 at reference qsq values
	  ff		= zfit_lib.zfit_ff(self.input_dict) 	# instantiate zfit_lib
	  ff.mpolep	= dd['polep']				# assign BCL poles for fp
	  ff.mpole0	= dd['pole0']				# assign BCL poles for f0
	  ff.tstar	= eval(dd['tstar'].replace('self','ff'))# compute and assign tsart
	  ff.t0		= eval(dd['t0'].replace('self','ff'))   # compute and assign t0
	  ff.tp		= eval(dd['tp'].replace('self','ff'))   # compute and assign tp
	  ff.tm		= eval(dd['tm'].replace('self','ff'))   # compute and assign tm
	  ff.Kp		= dd['Kp']
	  ff.K0		= dd['K0']
	  qsqp		= dd['qsqp']				# reference qsq for fp
	  qsq0		= dd['qsq0']				# reference qsq for f0
	  dd['fp'] 	= ff.zfit_BCL_p_fn(dd['bp'],qsqp)	# compute and assign fp
	  dd['f0'] 	= ff.zfit_BCL_0_fn(dd['b0'],qsq0)	# compute and assign f0
	  dd['Cff']	= ff.Cff_BCL(qsqp,qsq0,dd['Cp0'],dd['bp'],dd['b0']) # covariance matrix

	 # print the computed form factors at reference qsq vlaues, errors and correlation to table in latex format
	 self.print_nice_input(tag,len(qsqp),len(qsq0),np.r_[qsqp,qsq0],np.r_[dd['fp'],dd['f0']],dd['Cff'])
	 # print some info on data set to command line
	 print(self.separator) 
	 print('Added data %s'%tag)
	 if len(dd['qsqp'])>0:
	  print('qsq:	 ',' '.join(['%.1f'%qsql for qsql in dd['qsqp']]))
	  print('fp:	 ',' '.join(self.disperr(dd['fp'],np.sqrt(np.diag(dd['Cff']))[:len(dd['qsqp'])]))) 
	 if len(dd['qsq0'])>0:
	  print('qsq:	 ',' '.join(['%.1f'%qsql for qsql in dd['qsq0']]))
	  print('f0:	 ',' '.join(self.disperr(dd['f0'],np.sqrt(np.diag(dd['Cff']))[len(dd['qsqp']):]))) 
	 if dd['data type'] == 'BCL':
	  print('z-range:  %.3f <= z <= %.3f'%(ff.zfn((self.Mi-self.Mo)**2,ff.tstar,ff.t0),ff.zfn(0,ff.tstar,ff.t0)))
	  del ff
	 # append added data set to self.data
	 self.data.append(dd)

        ################################################################
	def make_design_matrix(self):
	 """
	   Construction of matrix Z -- Eq. (3.5) in the fitting paper
	   and combined the input data vector for the form factor
	   results and the global block-diagonal covariance matrix
	 """
	 # determine overall size
	 self.Z	= np.empty((0,self.Kp+self.K0-1),dtype='float')
	 self.y	= []
	 self.Cf= np.empty((0,0),dtype='float')

	 # loop over input data sets and construct matrix Z
	 # for each case and combined into block-diagonal form
	 for datl in self.data:	
	  Xpp = self.ff.zfit_BGL_p_fn([],datl['qsqp']   )
	  X00 = self.ff.zfit_BGL_0_fn([],datl['qsq0'],f0_constraint=True)
	  X0p = self.ff.zfit_BGL_p_fn([],datl['qsq0'],f0_constraint=True)
	  Xp0 = np.zeros((len(datl['qsqp']),self.K0-1))
	  X0  = np.r_['1',Xpp,Xp0]
	  X1  = np.r_['1',X0p,X00]
	  self.Z   = np.r_['0',self.Z,np.r_['0',X0,X1]]

	  # self.y holds form-factor input data	  
	  self.y   = np.r_[self.y,datl['fp'],datl['f0']]
	  # self.Cf holds covariance matrices of input data
	  # one diagonal-block for every data set
	  # In case of correlations between different data
	  # sets this matrix could be modified by hand 
	  # prior to calling run_BFF()
	  self.Cf  = scipy.linalg.block_diag(self.Cf,datl['Cff'])

        ################################################################
	def make_zizj(self,K,alpha):
	  """ 
	   Compute the <z^i|z^j> (Eq. (2.9) of fitting paper)
	  """ 
	  def res(i,j):
	   if i==j:
	    return alpha/np.pi
	   else:
	    return np.sin(alpha*(i-j))/(np.pi*(i-j))
	  M = np.array([[res(i,j) for i in range(K)] for j in range(K)])
	  return M

        ################################################################
	def make_M(self,Kp,K0,alpha):
	  """
	   This function returns the metric M for the Bayesian prior 
	   (cf. App. D of fitting paper)	
	  """
	  fp_pre0,_,_    = self.ff.zfit_BGL_p_pole(0)
	  f0_pre0,_,_ 	= self.ff.zfit_BGL_0_pole(0)	
	  Pratio 	= fp_pre0/f0_pre0

	  z		= self.ff.zfn(0,self.tstar,self.t0)
	  zpp		= np.array([[z**i*z**j for i in range(Kp)] for j in range(Kp)] )
	  z00		= np.array([[z**i*z**j for i in range(1,K0)] for j in range(1,K0)] )
	  zp0		= np.array([[z**i*z**j for i in range(Kp)] for j in range(1,K0)] )
	  z0p		= np.array([[z**i*z**j for i in range(1,K0)] for j in range(Kp)] )

	  Mp 		= self.make_zizj(self.Kp  ,self.angle)
	  M0		= self.make_zizj(self.K0  ,self.angle)
	  M00		= M0[0,0]
	  M0i		= M0[0,:]
	  M0bar		= M0[1:,1:]
	
	  K1		= +  Pratio**2 * M00 * zpp
	  K2		= -  Pratio    * M00 * z0p
	  K3		= -  Pratio    * M00 * zp0
	  K4		= +              M00 * z00
	  K5 		= +  Pratio    * np.array([[M0i[i]*z**j for i in range(1,K0)] for j in range(Kp)] )
	  K6 		= +  Pratio    * np.array([[M0i[i]*z**j for i in range(1,K0)] for j in range(Kp)] ).T
	  K7		= -              np.array([[M0i[i]*z**j for i in range(1,K0)] for j in range(1,K0)] ) 
	  K8		= -              np.array([[M0i[i]*z**j for i in range(1,K0)] for j in range(1,K0)] ).T
	  M		= scipy.linalg.block_diag(Mp,M0bar)
	  if 2==2:
	   M[:Kp,:Kp]	+=  K1
	   M[:Kp,Kp:]	+=  K2
	   M[Kp:,:Kp]	+=  K3
	   M[Kp:,Kp:]	+=  K4
	   M[:Kp,Kp:]	+=  K5
	   M[Kp:,:Kp]	+=  K6
	   M[Kp:,Kp:]	+=  K7
	   M[Kp:,Kp:]	+=  K8
	  return M

        ################################################################
	def run_BFF(self):
	 """
	  Core function that carries out the Bayesian inference (and 
	  as by product a frequentist fit for a given value of
	  (Kp,K0)
	  The procedure and algorithm are described in detail in 
	  Sec. 3.2 of the fitting paper
	 """

	 self.normterm_samples = []
	 # add (Kp,K0) to list of computed cases
	 self.Kcases.append([self.Kp,self.K0])
	 print(self.separator)
	 # overwrite defaults for (Kp,K0) in zfit_lib
	 self.ff.Kp	= self.Kp
	 self.ff.K0	= self.K0
	 # compute angle = arg[z(t+;t*,t0)] (see Sec. 2.2. of the fitting paper
	 self.angle	= np.angle(self.ff.zfn(self.tp,self.tstar,self.t0,epsilon=0j))
	 print('Angle on unit arc in z-plane: %.4f'%self.angle)


	 # initialise the Metric for the Bayesian prior
	 bigM		= self.make_M(self.Kp,self.K0,self.angle)
	 # initialise the matrix Mp_ij=<z^i|z^j> for the unitarity constraint
	 Mp 		= self.make_zizj(self.Kp  ,self.angle)
	 # compute the condition number of Mp	
	 # (check that Kp small enough such that condition number of Mp acceptable)
	 cond 		= np.linalg.cond(Mp)
	 if cond>1e7:
	  print('Condition number of metric close-to-singular: %.2e'%cond)
	  print('Consider being more modest in your choice of Kp and K0')
	  exit()
	 # repeat for M0
	 M0 		= self.make_zizj(self.K0  ,self.angle)
	 M0_red 	= np.delete(np.delete(M0,0,0),0,1)
	 M		= scipy.linalg.block_diag(Mp,M0)
	 cond 		= np.linalg.cond(M0)
	 cond_red	= np.linalg.cond(M0_red)
	 if (cond>1e7) or (cond_red>1e7):
	  print('Condition number of metric close-to-singular: %.2e'%cond,cond_red)
	  print('Consider being more modest in your choice of Kp and K0')
	  exit()

	 # initailise design matrix Z (see Sec. 3.1 of fitting paper)
	 self.make_design_matrix()

	 # get started with Bayesian inference
	 a0             = np.array([0.]*(self.Kp+self.K0-1))
	 # this is the metric for the Bayesian prior
	 invCa		= bigM/self.sigma**2
	 # construct (3.18) and (3.19) of fitting paper
	 invCy          = np.linalg.inv(self.Cf)
	 invtildeCy0    = np.dot(self.Z.T,np.dot(invCy,self.Z)) 
	 invtildeCy     = invtildeCy0 + invCa
	 tildeCy        = np.linalg.inv(invtildeCy)
	 dum            = np.dot(self.Z.T,np.dot(invCy,self.y))
	 print(self.separator)
	 print('Now doing (Kp,K0)=(%d,%d)'%(self.Kp,self.K0))

	 # if Ndof>=1 do frequentist fit
	 if self.Cf.shape[0]>self.Kp+self.K0-1:
	  print(self.separator)
	  print('Freuqentist fit possible:')
	  self.cov_frequ= np.linalg.inv(invtildeCy0)
	  self.y_frequ	= self.cov_frequ@dum
	  self.dy_frequ	= np.sqrt(np.diag(self.cov_frequ))
	  # compute chisq
	  delta		= (self.Z@self.y_frequ-self.y)
	  self.chisq	= delta@invCy@delta
	  self.Ndof	= len(self.y)-self.Kp-self.K0+1
	  self.pval	= self.mkchisqval(self.chisq,self.Ndof)
	  self.frequ_cases.append([self.Kp,self.K0])
	  print('Frequentist: chisq=%.2f, Ndof=%d, pval=%.2f%%'%(self.chisq/self.Ndof,self.Ndof,self.pval*100))

	  # reconstruct 0-component of a_0
	  a00           = np.array([self.ff.make_b0_BGL(self.y_frequ)])
	  da00		= self.dfunc(self.ff.make_b0_BGL,self.y_frequ,self.cov_frequ,[1])
	  y_frequ0	= np.r_[ self.y_frequ[:self.Kp],a00 , self.y_frequ[self.Kp:]]
	  d_frequ0	= np.r_[self.dy_frequ[:self.Kp],da00,self.dy_frequ[self.Kp:]]
	  print('Frequentist - BGL parameters:')
	  print(' '.join(self.disperr(y_frequ0,d_frequ0)))
	  print(self.separator)
	  
	  # Add results of frequentist fit to report 
	  self.report['results_frequ_%d_%d'%(self.Kp,self.K0)] = {
			'val0':		y_frequ0,
			'dval0':	d_frequ0,
			'sres':		self.disperr(y_frequ0,d_frequ0),
			'val':		self.y_frequ,
			'dval':		np.sqrt(np.diag(self.cov_frequ)),
			'cov':		self.cov_frequ,
			'chisq':	self.chisq,
			'Ndof':		self.Ndof,
			'pval':		self.pval}
	  self.bfrequ = 1

	 # Compute \tilde a
	 tmpa           = dum + np.dot(invCa,a0)
	 tildea         = np.dot(tildeCy,tmpa)

	 # start sampling
	 samples	= np.empty((0,self.Kp+self.K0))
	 length		= 0
	 N		= self.N
	 total_drawn	= 0
	 while (length<self.N): # run algorithm until desired # of samples reached
	  # draw atilde samples
	  samples0        = self.rnd.multivariate_normal(tildea,tildeCy,N)
	  total_drawn	 +=N
	  # reconstruct a_0,0
	  a00            = np.array([self.ff.make_b0_BGL(samples0)])
	  a0             = np.r_['1',a00.T,samples0[:,self.Kp:]]
	  ap             = np.r_['1',samples0[:,:self.Kp]]
	  # impose unitarity constraint
	  norm_ap 	= np.sum(np.multiply(ap.T,Mp@ap.T).T,1)
	  norm_a0 	= np.sum(np.multiply(a0.T,M0@a0.T).T,1)
	  ind           = np.where(( norm_ap <= 1 ) & ( norm_a0 <= 1 ))[0]
	  # purge results not compatible with unitarity cosntraint
	  samples0      = np.r_['1',ap,a0][ind,:]
	  s = 'unitarity constraint efficiency\t: %5.2f%%'%(len(ind)/ap.shape[0]*100)

 	  # accept/reject step to correct towards flat prior
	  a0_red	= np.delete(a0,0,1)
	  samples_red 	= np.delete(samples0,self.Kp,1)
	  normterm	= np.sum(np.multiply(samples_red.T,bigM@samples_red.T).T,1)
	  Ns             = samples0.shape[0]
	  c              = np.exp(-0.5*(       2.0/self.sigma**2) )
	  priorterm      = c/(np.exp(-0.5*(normterm/self.sigma**2)))
	  self.normterm_samples.append(np.max(normterm))

	  if (priorterm > 1).any(): # sanity check -- exit of accept/reject-step normalisation violated
				    # this should really never happen	
	   print('priorterm not correctly normalised',min(priorterm),max(priorterm),max(normterm))
	   exit()
	  # draw flat random number
	  r              = np.random.rand(Ns)
	  # accept/reject with probability priorterm
	  ind            = np.where(r<=priorterm)[0]
	  tmp   	 = samples0[ind,:]
	  # accumulated samples	
	  samples	 = np.r_['0',samples,tmp]
	  length	 = samples.shape[0]
	  if length>0:
	   efficiency	 = length/total_drawn
	   still_to_draw = self.N-length
	   N		 = np.min([int(1e7),int(np.ceil(still_to_draw/efficiency))])
	   if N>0:
 	    print('estimate of required samples	: %d'%(N))
	 print(s)
	 print('accept/reject step efficiency\t: %5.2f%%'%(efficiency*100))

	 # Wrap things up
	 self.samples	= samples[:self.N,:]	
	 # This is cute -- I can rotate change the basis of the BGL coefficients
	 # such that the norm-squared of the undetermined coefficients has 
	 # variance like a flat random numbers in a K-dimensional unit ball
	 # for which the expectation is 1/(K+2) -- nice numerical check of 
	 # implementation. I keep it in the code for now.
	 sqrtM		= scipy.linalg.sqrtm(M)
	 self.Ua	= pt.particle()
	 self.Ua.samples= (sqrtM@self.samples.T).T
	 # assign final results
	 self.val	= np.mean(self.samples,axis=0)
	 self.dval	= np.std(self.samples,axis=0)
	 self.covariance= np.cov(self.samples.T)
	 # print coefficients with errors
	 print(self.separator)
	 print('Bayesian inference result for BGL params:')
	 print(' '.join(self.disperr(self.val,self.dval)))

	 # now add observables to report
	 self.report['results_%d_%d'%(self.Kp,self.K0)] = {
	      'val':		self.val,
	      'dval':		self.dval,
	      'Uval':		self.Ua.bs_val(),
	      'dUval':	self.Ua.bs_dval(),
	      'cov':		self.covariance,
	      'samples': 	self.samples,
	      'Uasamples': 	self.Ua.samples,
	      }
	 # if experimental input is provided, 
	 # CKM matrix elements have been computed and we 
	 # add results to report

	 return self.samples






	##############################################################
	def make_plots(self):
	 """
	  Produce nicely formatted plots of:
	   - form factor and BGL param vs. q^2 and vs. z
	   - triangle correlation plots (if self.triangle=1)
	 """
	 fig,ax 	= plt.subplots(1,2,figsize=(10,5))
	 l0		= [] # these lists will hold the plot pointer later used to generated legends
	 l1		= []
	 l2		= []
	 l3		= []
	 sl		= []
	 # plot form factors at reference q^2 values,
	 # once vs. q^2 and once vs. z(q^@)
	 for datl in self.data:
	  qsqp		= datl['qsqp']
	  qsq0		= datl['qsq0']
	  fp		= datl['fp']
	  f0		= datl['f0']
	  dd		= np.sqrt(np.diag(datl['Cff']))
	  fp_pre,_,_ 	= self.ff.zfit_BGL_p_pole(qsqp)
	  f0_pre,_,_ 	= self.ff.zfit_BGL_0_pole(qsq0)	
	  fp_pre0,_,_ 	= self.ff.zfit_BGL_p_pole(0)
	  f0_pre0,_,_ 	= self.ff.zfit_BGL_0_pole(0)	
	  if len(qsqp)>0:
	   l0.append(ax[0].errorbar(qsqp,fp,dd[:len(fp)],fmt='o',markerfacecolor="None",capsize=5,lw=1.5))
	   l2.append(ax[1].errorbar(self.ff.zfn(qsqp,self.tstar,self.t0),fp/fp_pre*fp_pre0,dd[:len(fp)]/fp_pre*fp_pre0,
					fmt='o',markerfacecolor="None",capsize=5))
	  if len(qsq0)>0:
	   l1.append(ax[0].errorbar(qsq0,f0,dd[len(fp):],fmt='s',markerfacecolor="None",capsize=5,lw=1.5))
	   l3.append(ax[1].errorbar(self.ff.zfn(qsq0,self.tstar,self.t0),f0/f0_pre*f0_pre0,dd[len(fp):]/f0_pre*f0_pre0,
					fmt='s',markerfacecolor="None",capsize=5))
	 # add legend
	 ax[0].legend(l0+l1,
		[('$f_+(q^2) {\\rm %s}$'%ii).replace(' ','\,').replace(' lat','') for ii in self.labels]+
		[('$f_0(q^2) {\\rm %s}$'%ii).replace(' ','\,').replace(' lat','') for ii in self.labels],
		prop={'family':'serif','size':self.legendfontsize}, frameon=False)
	 # Now plot BGL parameterisation with error bands
	 qsq 		= np.linspace(0,self.tm,100)
	 fp 		= self.ff.zfit_BGL_p_fn(self.val[:self.Kp],qsq)
	 f0 		= self.ff.zfit_BGL_0_fn(self.val[self.Kp:],qsq)	
	 ffp_lambda 	= lambda a:self.ff.zfit_BGL_p_fn(a,qsq)
	 ff0_lambda 	= lambda a:self.ff.zfit_BGL_0_fn(a,qsq)
	 fp_pre,_,_ 	= self.ff.zfit_BGL_p_pole(qsq)
	 f0_pre,_,_ 	= self.ff.zfit_BGL_0_pole(qsq)	
	 dfp 		= self.dfunc(ffp_lambda,self.val[:self.Kp],self.covariance[:self.Kp,:self.Kp],qsq)
	 df0 		= self.dfunc(ff0_lambda,self.val[self.Kp:],self.covariance[self.Kp:,self.Kp:],qsq)
	 z  		= self.ff.zfn(qsq,self.tstar,self.t0)
	 ax[0].plot(qsq,fp,'b-')
	 ax[0].plot(qsq,f0,'r-')
	 ax[1].plot(z,fp/fp_pre*fp_pre0,'b-')
	 ax[1].plot(z,f0/f0_pre*f0_pre0,'r-')
	 ax[0].fill_between(qsq,fp+dfp,fp-dfp,color='b',alpha=.2 )
	 ax[0].fill_between(qsq,f0+df0,f0-df0,color='r',alpha=.2 )
	 ax[1].fill_between(z,(fp+dfp)/fp_pre*fp_pre0,(fp-dfp)/fp_pre*fp_pre0,color='b',alpha=.2 )
	 ax[1].fill_between(z,(f0+df0)/f0_pre*f0_pre0,(f0-df0)/f0_pre*f0_pre0,color='r',alpha=.2 )


	 # wrap things up and save to disk
	 ax[0].set_xlabel('$q^2\,[{\\rm GeV}^2]$')
	 ax[0].set_ylabel('$f_X(q^2)$')
	 ax[1].set_xlabel('$z(q^2)$')
	 ax[1].set_ylabel('$f_X(q^2)\phi_X(q^2)B_X(q^2)/(\phi_X(0)B_X(0))$')
	 fig.tight_layout()
	 fig.savefig('Report/plots/plot_%d_%d.pdf'%(self.Kp,self.K0)) 
	 fig.clf()
	 
	 # make triangle correlation plots for BGL coefficients
	 if self.triangle == 1:
	  g = plots.get_subplot_plotter()
	  samples_l = getdist.MCSamples(samples=self.samples)
	  g.triangle_plot
	  g.triangle_plot(samples_l,filled=True)
	  g.export('Report/plots/triangle_%d_%d.pdf'%(self.Kp,self.K0))
	  plt.clf()
	 # make triangle correlation plots for rotated BGL coefficients
	 if self.triangle == 1:
	  g = plots.get_subplot_plotter()
	  samples_l = getdist.MCSamples(samples=self.Ua.samples)
	  g.triangle_plot
	  g.triangle_plot(samples_l,filled=True)
	  g.export('Report/plots/triangle_Ua_%d_%d.pdf'%(self.Kp,self.K0))
	  plt.clf()


	##############################################################
	def make_tables(self):
	 """
	  Here we generate Latex tables for main results
	 """
	 #################################
	 # Bayesian fit results
	 #################################
	 wfp=open('Report/tables/BFF_fits_p.txt','w')
	 wf0=open('Report/tables/BFF_fits_0.txt','w')
	 K = self.Kp+self.K0
	 Kps = []
	 K0s = []
	 for cases in self.Kcases:
	  Kps.append(cases[0])
	  K0s.append(cases[1])
	 maxKp	= max(Kps)
	 maxK0	= max(K0s)
	 maxKp0 = min([10,max([max(Kps),max(K0s)])])
	 for i,(f,sf) in enumerate(zip([wfp,wf0],['+','0'])):
	  f.write('\\begin{tabular}{l@{\hspace{1mm}}llllllllllllllllllllllllllllllllllllllllllllllllll}\n')
	  f.write('\hline\hline\n')
	  f.write('$K_+$&$K_0$&'+'&'.join(['\multicolumn{1}{c}{$a_{'+sf+',%d}$}'%i for i in range(maxKp0)]) +'&\\\\\n')
	  f.write('\hline\n')
	 for cases in self.Kcases:
	  Kp	= cases[0]
	  K0	= cases[1]
	  rep	= self.report['results_%d_%d'%(Kp,K0)]
	  val	= rep['val']
	  dval	= rep['dval']
	  if (Kp<=10):
	   wfp.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[:Kp],dval[:Kp]))+' '.join(['&-']*(10-Kp)) +'&\\\\\n')
	  else:
	   wfp.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[:10],dval[:10])) +'&\\\\\n')
	  if (K0<=10):
	   wf0.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[Kp:],dval[Kp:]))+' '.join(['&-']*(10-K0)) +'&\\\\\n')
	  else:
	   wf0.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[Kp:Kp+10],dval[Kp:Kp+10])) +'&\\\\\n')
	 for f,sf in zip([wfp,wf0],['+','0']):
	  f.write('\hline\hline\\\\\n')
	  f.write('\end{tabular}\n')
	  f.close()
	 #################################
	 # Bayesian fit results for rotated coefficients
	 #################################
	 wfp=open('Report/tables/BFF_fits_p_rotated.txt','w')
	 wf0=open('Report/tables/BFF_fits_0_rotated.txt','w')
	 K = self.Kp+self.K0
	 Kps = []
	 K0s = []
	 for cases in self.Kcases:
	  Kps.append(cases[0])
	  K0s.append(cases[1])
	 maxKp	= max(Kps)
	 maxK0	= max(K0s)
	 maxKp0 = min([10,max([max(Kps),max(K0s)])])
	 for i,(f,sf) in enumerate(zip([wfp,wf0],['+','0'])):
	  f.write('\\begin{tabular}{l@{\hspace{1mm}}llllllllllllllllllllllllllllllllllllllllllllllllll}\n')
	  f.write('\hline\hline\n')
	  f.write('$K_+$&$K_0$&'+'&'.join(['\multicolumn{1}{c}{$a_{'+sf+',%d}$}'%i for i in range(maxKp0)]) +'&\\\\\n')
	  f.write('\hline\n')
	 for cases in self.Kcases:
	  Kp	= cases[0]
	  K0	= cases[1]
	  rep	= self.report['results_%d_%d'%(Kp,K0)]
	  val	= rep['Uval']
	  dval	= rep['dUval']
	  if (Kp<=10):
	   wfp.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[:Kp],dval[:Kp]))+' '.join(['&-']*(10-Kp)) +'&\\\\\n')
	  else:
	   wfp.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[:10],dval[:10])) +'&\\\\\n')
	  if (K0<=10):
	   wf0.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[Kp:],dval[Kp:]))+' '.join(['&-']*(10-K0)) +'&\\\\\n')
	  else:
	   wf0.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[Kp:Kp+10],dval[Kp:Kp+10])) +'&\\\\\n')
	 for f,sf in zip([wfp,wf0],['+','0']):
	  f.write('\hline\hline\\\\\n')
	  f.write('\end{tabular}\n')
	  f.close()
	 #################################
	 # Frequentist fit results 
	 #################################
	 if self.bfrequ: 
	  # Frequentist fit results
	  wfp=open('Report/tables/BFF_frequ_fits_p.txt','w')
	  wf0=open('Report/tables/BFF_frequ_fits_0.txt','w')
	  # max number of params that can be determined:
	  fcp = []
	  fc0 = []
	  for fc in self.frequ_cases:
	   fcp.append(fc[0]) 
	   fc0.append(fc[0]) 
	  fcp = max(fcp)
	  fc0 = max(fc0)
	  Np_max = fcp+fc0
	  K = self.Kp+self.K0
	  Kps = []
	  K0s = []
	  for cases in self.Kcases:
	   Kps.append(cases[0])
	   K0s.append(cases[1])
	  maxKp0 = max([max(Kps),max(K0s)])
	  for i,(f,sf,fc) in enumerate(zip([wfp,wf0],['+','0'],[fcp,fc0])):
	   f.write('\\begin{tabular}{l@{\hspace{1mm}}llllllllllllllllllllllllllllllllllllllllllllllllll}\n')
	   f.write('\hline\hline\n')
	   f.write('$K_+$&$K_0$&'+'&'.join(['\multicolumn{1}{c}{$a_{'+sf+',%d}$}'%i for i in range(fc)]) +\
							'&$p$&$\chi^2/N_{\\rm dof}$&$N_{\\rm dof}$\\\\\n')
	   f.write('\hline\n')
	  for cases in self.Kcases:
	   Kp	= cases[0]
	   K0	= cases[1]
	   if 'results_frequ_%d_%d'%(Kp,K0) in self.report.keys():
	    rep	= self.report['results_frequ_%d_%d'%(Kp,K0)]
	    val		= rep['val0']
	    dval	= rep['dval0']
	    pval	= rep['pval']
	    Ndof	= rep['Ndof']
	    chisq	= rep['chisq']
	    wfp.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[:Kp],dval[:Kp]))+' '.join(['&-']*(fcp-Kp)) +\
					'& %.2f& %.2f&%d&\\\\\n'%(pval,chisq/Ndof,Ndof))
	    wf0.write('%d&%d&'%(Kp,K0)+'&'.join(self.disperr(val[Kp:],dval[Kp:]))+' '.join(['&-']*(fc0-K0)) +\
					'& %.2f& %.2f&%d&\\\\\n'%(pval,chisq/Ndof,Ndof))
	  for f,sf in zip([wfp,wf0],['+','0']):
	   f.write('\hline\hline\\\\\n')
	   f.write('\end{tabular}\n')
	   f.close()
	##############################################################
	def print_nice_input(self,tag,Np,N0,qsq,ff,cov):
	 """
	   does what it says
	 """
	 f=open('Report/tables/input_'+tag.replace(' ','_')+'.txt','w')
	 self.print_table_header(f,''.join(['l|']+['l']*Np+['|']+['l'*N0]))
	 f.write('&\multicolumn{'+str(Np)+'}{c|}{$f_+$}&\multicolumn{'+str(N0)+'}{c}{$f_0$}\\\\\n')
	 f.write('\hline\hline\n')
	 f.write('$q^2/\gev$&'+'& '.join(['%.1f'%qsq for qsq in qsq])+'\\\\\n')
	 dff = np.sqrt(np.diag(cov))
	 f.write('$f$&'      +'& '.join(['%.4f'%ff  for  ff in  ff])+'\\\\\n')
	 f.write('$\delta f$&'+'& '.join(['%.4f'%dff for dff in dff])+'\\\\\n')
	 f.write('\hline\hline\n')
	 for ii,line in enumerate(self.make_corr(cov)):
	  sext = '\\\\\n'
	  if ii==Np-1:
	   sext = '\\\\[-2.0ex]\n'
	  if ii==Np:
	   f.write('&\cline{1-'+str(Np+N0)+'}\\\\[-3.7ex]\n')
	  f.write('&'+'& '.join(['%.4f'%c for c in line])+sext)
	 f.write('\hline\hline\n')
	 self.print_table_footer(f)
	 f.close() 
	 
	##############################################################
	def make_report(self,tag=''):
	  """
	   Create a Latex report summarising all main results
	  """
	  # first dump results to hdf5
	  f = h5py.File('Report/h5dump.h5','w')
	  # dump input_dict
	  for sitem in self.input_dict.keys():
	    if 'outer' in sitem:
	     dum = self.input_dict[sitem]
	     f.create_dataset('input_dict/'+sitem,data=np.r_[dum[0],eval(dum[1]),dum[2],dum[3]])
	    elif 'experimental_input' in sitem:
	     continue
	    else:
	     f.create_dataset('input_dict/'+sitem,data=self.input_dict[sitem])
	  # dump added_data
	  for dataset in self.data:
	   for dd in dataset.keys():
	    print(dd,dataset)
	    print('data_set/'+dataset['tag']+'/'+dd)
	    f.create_dataset('data_set/'+dataset['tag']+'/'+dd,data=dataset[dd])
	  for scase in self.report.keys():
	   dat = self.report[scase]
	   for sdat in dat.keys():
	    f.create_dataset(scase+'/'+sdat,data=dat[sdat])
	  f.close

	  self.make_tables()
	  if tag=='':
	   fn = 'BFF_report.tex'
	  else:
	   fn = 'BFF_report_'+tag+'.tex'
	  f=open('Report/'+fn,'w')
	  f.write('\documentclass{article}\n')
	  f.write('\pagestyle{empty}\n')
	  f.write('\\usepackage{hyperref}\n')
	  f.write('\hypersetup{\n')
	  f.write('    colorlinks,\n')
	  f.write('    citecolor=black,\n')
	  f.write('    filecolor=black,\n')
	  f.write('    linkcolor=black,\n')
	  f.write('    urlcolor=black\n')
	  f.write('}\n')
	  f.write('\\usepackage{graphicx}')
	  f.write('\\title{Bayesian form factor fit -- report}')
	  f.write('\\begin{document}')
	  f.write('\maketitle')
	  f.write('\\tableofcontents')
	  f.write('\\newpage')
	  #
	  f.write('\section{Code}\n')

	  repo = git.Repo(search_parent_directories=True)
	  sha = repo.head.object.hexsha
	  f.write('hash of last BFF commit: '+sha)
	  #
	  f.write('\section{$z$-fit setup}\n')
	  for key, value in self.input_dict.items(): 
	   f.write('\\verb|%-20s|: %s\\\\\n' % (key, value))
	  #
	  f.write('\section{Input data}\n')
	  for tag in self.tags: 
	   f.write('\\verb|%-20s|: %s\\\\\n' % ('Tag',tag))
	   for key1, value1 in self.data_dict[tag].items(): 
	    if key1 == 'fp':
	     fp = self.data_dict[tag][key1] 
	    if key1 == 'f0':
	     f0 = self.data_dict[tag][key1] 
	    if key1 == 'Cff':
	     dd = np.sqrt(np.diag(self.data_dict[tag][key1] ))
	     f.write('\\verb|%-20s|: %d,%d\\\\\n' % ('Np,N0',len(fp),len(f0)))
	     f.write('\\verb|%-20s|: %s\\\\\n' % ('f',' '.join(self.disperr(np.r_[fp,f0],dd))))
	     NN	= value1.shape
	     corr = self.make_corr(value1)
	     f.write('\\verb|%-20s|: %.2e\\\\\n' % ('corr cond. nr.',np.linalg.cond(corr)))
	     for line in corr:
	      f.write('\\verb|%-20s|: %s\\\\\n' % ('corr',' '.join(['%.4f'%ii for ii in  line])))
	  #
	  f.write('\section{Results for BGL analysis}\n')
	  f.write('\subsection{$\chi^2$ fit without unitarity constraint}\n')
	  f.write('\\begin{center}')
	  f.write('\\tiny')
	  f.write('\input{tables/BFF_frequ_fits_p.txt}\n')
	  f.write('\input{tables/BFF_frequ_fits_0.txt}\n')
	  f.write('\end{center}')
	  #
	  f.write('\subsection{BGL coefficients}\n')
	  f.write('\\begin{center}')
	  f.write('\\tiny')
	  f.write('\input{tables//BFF_fits_p.txt}\n')
	  f.write('\input{tables//BFF_fits_0.txt}\n')
	  f.write('\end{center}\n')
	  #
	  f.write('\subsection{BGL coefficients -- rotated}\n')
	  f.write('\\begin{center}')
	  f.write('\\tiny')
	  f.write('\input{tables//BFF_fits_p_rotated.txt}\n')
	  f.write('\input{tables//BFF_fits_0_rotated.txt}\n')
	  f.write('\end{center}\n')
	  #
	  f.write('\subsection{Form-factor plots}\n')
	  for cases in self.Kcases:
	   f.write('\subsubsection{$(K_+,K_0)$=(%d,%d)}\n'%(cases[0],cases[1]))
	   f.write('\\begin{center}\n')
	   f.write('\includegraphics[width=10cm]{{plots/plot_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
	   f.write('\\end{center}\n')
	  if self.triangle == 1:
	   f.write('\subsection{Triangle  plots}\n')
	   for cases in self.Kcases:
	    f.write('\subsubsection{$(K_+,K_0)$=(%d,%d)}\n'%(cases[0],cases[1]))
	    f.write('\\begin{center}\n')
	    f.write('\includegraphics[width=10cm]{{plots/triangle_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
	    f.write('\\end{center}\n')
	   f.write('\subsection{Triangle plots rotated}\n')
	   for cases in self.Kcases:
	    f.write('\subsubsection{$(K_+,K_0)$=(%d,%d)}\n'%(cases[0],cases[1]))
	    f.write('\\begin{center}\n')
	    f.write('\includegraphics[width=10cm]{{plots/triangle_Ua_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
	    f.write('\\end{center}\n')
	  f.write('\end{document}\n')
	  f.close()
	  os.system('cd Report;pdflatex '+fn)
	  os.system('cd Report;pdflatex '+fn)

	##############################################################
	##############################################################
	#  SOME MISC HELPER FUNCTINOS
	##############################################################
	##############################################################
	def symmetrise(self,v,dstatm,dstatp,dsystm,dsystp):	
	   """
	     symmetrise central value and errors in case of asymmetric error bars
	   """
	   v		= v + 0.5*(dstatp-dstatm+dsystp-dsystm)
	   dstat	= 0.5*(dstatm+dstatp)
	   dsyst	= 0.5*(dsystm+dsystp)
	   return v,dstat,dsyst
        ################################################################
	def dfunc(self,fn,alpha,cov,*args):#qsq,tp,t0,mpole,K):
	  """
	    This is a helper function that numerically evaluates
	    a correlated error propagation. It is mainly used 
	    to generated error bands for plots
	  """
	  if args!=[]:
	   if isinstance(args[0],list):
	    ll = len(args[0])
	   else:
	    ll = args[0].shape[0]
	   Df = np.zeros((len(alpha),ll)) # args[0] is qsq here
	  else: 
	   Df = np.zeros((len(alpha),1)) # args[0] is qsq here
	  for i in range(len(alpha)):
	   eps = np.zeros((len(alpha),))
	   eps[i] =  0.01*alpha[i]
	   if args==[]:
	    dfp= fn(alpha+eps,*args)
	    dfm= fn(alpha-eps,*args)
	   else:
	    dfp= fn(alpha+eps)
	    dfm= fn(alpha-eps)
	   if eps[i]==0.:
	    Df[i,:] = 0.
	   else:
	    Df[i,:] = 0.5*(dfp-dfm)/eps[i]
	  df=0
	  for i in range(len(alpha)):
	   for j in range(len(alpha)):
	    df+=Df[i,:]*Df[j,:]*cov[i,j]
	  return np.sqrt(df)

	##############################################################
	def disperr(self,val,dval):
	    """
	     formatted print result with error
	    """
	    n=len(val)
	    if n!=len(dval):
	     print("val and dval must have the same length!")
	     print(val,dval)
	     print("exiting")
	     exit()
	    dig=2
	    res = n*['']
	    for i in range(n):
	     if dval[i] == 0. and val[i] == 0.:
	      res[i]     = "0"
	     elif np.isnan(val[i]) or np.isnan(dval[i]):
	      res[i]     = "nan"
	     elif dval[i] == 0. and val[i] != 0.:
	      value      = "%d" % val[i]
	      res[i]     = value
	     elif dval[i] < 1:
	      location   = int(np.floor(np.log10(dval[i])))
	      append_err = "("+str(int(np.round(dval[i]*10**(-location+dig-1))))+")"
	      if abs(val[i])<1e-100:
	       val[i]=0.
	       location=1
	      valformat  = "%."+str(-location+dig-1)+"f"
	      sval       = valformat % val[i]
	      res[i]     = sval +append_err
	     elif dval[i]>=1:
	      digits     = min(0,int(np.ceil(np.log10(dval[i]))-1))+1
	      error      = np.around(dval[i],digits)
	      value      = np.around(val[i],digits)
	      serr       = "%."+str(digits)+"f(%."+str(digits)+"f)"
	      serr       = serr%(value,error)
	      res[i]     = serr#str(value)+"("+str(error)+")"
	     else:
	      digits     = max(0,int(np.ceil(np.log10(dval[i]))-1))
	      error      = int(round(dval[i]/10**digits)*10**digits)
	      value      = round(val[i]/10**digits)*10**digits
	      res[i]     = str(value)+"("+str(error)+")"
	    return res
        ################################################################
	def mkchisqval(self,chisq,Ndof):
	 """
	  takes result of leastsq as input and returns
	   p-value,
	   chi^2/Ndof
	   Ndof
	 """
	 pv    = scipy.special.gammaincc(Ndof/2.,chisq/2.)
	 return pv
	##############################################################
	def print_table_header(self,f,format_string='lllll'):
	 """
	  helper function for Latex tables
	 """
	 f.write('\\begin{tabular}{'+format_string+'}\n')
	 f.write('\hline\hline\n')
	##############################################################
	def print_table_footer(self,f):
	 """
	  helper function for Latex tables
	 """
	 f.write('\end{tabular}\n')
	 f.close()
	##############################################################
	def make_corr(self,cov):
	  """
	   helper function computing correlation matrix from covariance matrix
	  """
	  v 	 = np.diag(1./np.sqrt(np.diag(cov)))
	  ncf 	 = np.dot(cov,v)
	  return np.dot(v,ncf)
	##############################################################
	def improved_R(self,a,ml,qsq1,qsq2):
	  """
	    This compute the improved R ratio
	  """
	  qsqmax = (self.Mo-self.Mi)**2
	  num0   = scipy.integrate.quad(self.improved_R_kernel_0_num,qsq1,qsq2,args=(a))
	  denom0 = scipy.integrate.quad(self.improved_R_kernel_0_den,qsq1,qsq2,args=(a,ml))
	  return num0[0]/denom0[0]
	##############################################################
	def improved_R_kernel_0_num(self,qsq,a):
	  """
	    This is the integral kernel for the numerator of the improved R ratio
	  """
	  ksq 		= self.ff.ksq_qsq(qsq)
	  fp 		= self.ff.zfit_BGL_p_fn(a[:self.Kp],qsq)
	  f0 		= self.ff.zfit_BGL_0_fn(a[self.Kp:],qsq)
	  Phi 		= np.sqrt(ksq)
	  omega_tau 	= (1.-pc.mtau**2/qsq)**2*(1.+pc.mtau**2/2./qsq)
	  FVsq 		= ksq*fp**2
	  FSsq 		= 3./4.*pc.mtau**2/(pc.mtau**2+2*qsq)*(self.Mi**2-self.Mo**2)**2/self.Mi**2*f0**2
	  return Phi*omega_tau*(FVsq+FSsq)#Phi*omega_l*FSsq

	##############################################################
	def improved_R_kernel_0_den(self,qsq,a,ml):
	  """
	    This is the integral kernel for the denominator of the improved R ratio
	  """
	  ksq 		= self.ff.ksq_qsq(qsq)
	  ap 		= a[:self.Kp]
	  a0 		= a[self.Kp:]
	  fp 		= self.ff.zfit_BGL_p_fn(ap,qsq)
	  f0 		= self.ff.zfit_BGL_0_fn(a0,qsq)
	  Phi 		= np.sqrt(ksq)
	  mtau		= pc.mtau
	  omega_l 	= (1.-ml**2/qsq)**2*(1.+ml**2/2./qsq)
	  omega_tau 	= (1.-mtau**2/qsq)**2*(1.+mtau**2/2./qsq)
	  FVsq 		= ksq*fp**2
	  FSsq 		= 3./4.*mtau**2/(mtau**2+2*qsq)*(self.Mi**2-self.Mi**2)**2/self.Mi**2*f0**2
	  return Phi*omega_tau/omega_l*(FVsq+FSsq)#Phi*omega_l*FSsq

	##############################################################
	def reduced_gamma_0_BGL(self,qsq,ml,a):
	  """
	   This is hte differential decay rate without normalisation factor
	  """
	  ksq 		= self.ff.ksq_qsq(qsq) 
	  E_out		= np.sqrt(self.Mo**2+ksq)
	  ap 		= a[:self.Kp]
	  a0 		= a[self.Kp:]
	  # relevant part of differential semileptonic decay rate
	  fp 		= self.ff.zfit_BGL_p_fn(ap,qsq)
	  f0 		= self.ff.zfit_BGL_0_fn(a0,qsq)
	  prefac 	= (qsq-ml**2)**2/qsq**2*np.sqrt(E_out**2-self.Mo**2)/self.Mi**2
	  resp		= (1+ml**2/2./qsq)*self.Mi**2*(E_out**2-self.Mo**2)*fp**2
	  res0		= 3*ml**2/8./qsq*(self.Mi**2-self.Mo**2)**2*f0**2
	  return prefac*(resp+res0)

	##############################################################
	def AFBker_BGL(self,qsq,ml,a):
	  """
	   This is kernel for the forward-backward asymmetry
	  """
	  ksq 		= self.ff.ksq_qsq(qsq) 
	  E_out		= np.sqrt(self.Mo**2+ksq)
	  ap 		= a[:self.Kp]
	  a0 		= a[self.Kp:]
	  fp 		= self.ff.zfit_BGL_p_fn(ap,qsq)
	  f0 		= self.ff.zfit_BGL_0_fn(a0,qsq)
	  return 24/32 * 1./self.Mi * (1 - ml**2/qsq)**2 * ksq * ml**2/qsq * (self.Mi**2 - self.Mo**2) * fp*f0

	##############################################################
	def Apolker_BGL(self,qsq,ml,a):
	  """
	   This is kernel for the polarisation asymmetry
	  """
	  ksq 		= self.ff.ksq_qsq(qsq) 
	  E_out		= np.sqrt(self.Mo**2+ksq)
	  ap 		= a[:self.Kp]
	  a0 		= a[self.Kp:]
	  fp 		= self.ff.zfit_BGL_p_fn(ap,qsq)
	  f0 		= self.ff.zfit_BGL_0_fn(a0,qsq)
	  LH		= np.sqrt(ksq)**3 *             (1 - ml**2/qsq)**2 * fp**2
	  RH 		= np.sqrt(ksq)    * ml**2/qsq * (1 - ml**2/qsq)**2 * \
				(3./8. * (self.Mi**2 - self.Mo**2)**2/self.Mi**2 * f0**2 +
			         1./2. * ksq * fp**2)
	  return LH-RH

	##############################################################
	def gen_R_AFB_BGL(self,a,ml1,ml2):
	  """
	   This routine integrates the decay width and the asymmetry and 
	   multiplies the normalisation factor
	  """
	  eta_EW   = a[-1]
	  a	   = a[:self.Kp+self.K0]
	  qsqmax   = self.tm
	  numAFB1  = scipy.integrate.quad(self.AFBker_BGL ,ml1**2,qsqmax,args=(ml1,a))
	  numAFB2  = scipy.integrate.quad(self.AFBker_BGL ,ml2**2,qsqmax,args=(ml2,a))
	  numApol1 = scipy.integrate.quad(self.Apolker_BGL,ml1**2,qsqmax,args=(ml1,a))
	  numApol2 = scipy.integrate.quad(self.Apolker_BGL,ml2**2,qsqmax,args=(ml2,a))
	  Gamma1   = scipy.integrate.quad(self.reduced_gamma_0_BGL,ml1**2,qsqmax,args=(ml1,a))
	  Gamma2   = scipy.integrate.quad(self.reduced_gamma_0_BGL,ml2**2,qsqmax,args=(ml2,a))
	  N 	   = eta_EW*pc.GF**2./(24*np.pi**3)/pc.hbar
	  return Gamma1[0]/Gamma2[0],numAFB1[0]/Gamma1[0],numAFB2[0]/Gamma2[0],numAFB1[0]*N,numAFB2[0]*N,\
			Gamma1[0]*N,Gamma2[0]*N,\
			numApol1[0]/Gamma1[0],numApol2[0]/Gamma2[0],numApol1[0]*N,numApol2[0]*N

	##############################################################
	def fVCKM(self,alpha,*args):
	   """
	     compute VCKM for given set of BGL parameters
	   """
	   Kp		= args[0]	
	   K0		= args[1]	
	   qsqmin	= args[2]	
	   qsqmax	= args[3]	
	   other_input 	= alpha[Kp+K0:]
	   RBF		= other_input[0]
	   BBstoDs	= other_input[1]
	   tauBs	= other_input[2]
	   eta_EW	= other_input[3]
	   res = self.BstoK_BGL_VCKM(alpha[:Kp+K0],pc.mmu,qsqmin,qsqmax,RBF,BBstoDs,tauBs,eta_EW)
	   return np.array([res])

	##############################################################
	def BstoK_BGL_VCKM(self,alpha,mmu,qsqmin,qsqmax,RBF,BBstoDs,tauBs,eta_EW):
	   """
	     integrate the decay width for the CKM-matrix determination
	   """
	   alpha	= alpha[:self.Kp+self.K0]
	   Gamma 	= scipy.integrate.quad(self.reduced_gamma_0_BGL,qsqmin,qsqmax,args=(mmu,alpha))[0]
	   N 		= eta_EW*pc.GF**2./(24*np.pi**3)/pc.hbar
	   Deltaxi	= N*Gamma
	   return (np.sqrt(RBF*BBstoDs/tauBs/Deltaxi))


