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
import lib.BFF_lib as Bff
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
import particles.particles as pt
import h5py
# The multiprocessing support is currently experimental. 
try:
    import multiprocessing
    MULTIPROCESSING = True
except ImportError:
    print('Module multiprocessing not found -- proceed with out it')
    print('Consider installing the module for better performance')
    MULTIPROCESSING = False
    pass



###############################################################################
# multprocessing aux functions -- not supported at the moment, under development
###############################################################################
def par_kernel(chunk):
    f		= chunk[1]
    args	= chunk[2]
    res 	= np.apply_along_axis(f,1,chunk[0],*args)
    return res
# apply_along_axis for multprocessing -- not supported, under development
def my_apply_along_axis(f,samples,*args,Nproc=6,parallel=True): 
    if (MULTIPROCESSING == False) or (parallel==False):
     # use single-core version
     return np.apply_along_axis(f,1,samples,*args)
    else:
     # parallelise over Nproc cores
     # Chunks for the mapping (only a few chunks):
     chunks 	= [(sub_arr,f,args) for sub_arr in np.array_split(samples, Nproc)]
     pool 	= multiprocessing.get_context("fork").Pool(Nproc)
     individual_results = pool.map(par_kernel, chunks)
     # Freeing the workers:
     pool.close()
     pool.join()
     return np.concatenate(individual_results)


class Bff_pheno(Bff.Bff):
        ################################################################
	def __init__(self,input_dict,large_t=False,large_t_ibase=0):
	 super().__init__(input_dict,large_t=large_t,large_t_ibase=large_t_ibase) # supercharge with base class Bff
	 self.report_report_plots=False
	 self.report_report_observables=False
	def make_pheno_plots(self):
	 self.report_report_plots=True
	 # pheno plots
	 if not 'experimental_input' in self.input_dict.keys():
	  print("make_pheno_plots: no experimental_input provided")
	  exit()
	 else:
	  ###################################
	  # make differential-decay-rate plots
	  colors		= ['r','b']
	  N 		= pc.eta_EW*pc.GF**2./(24*np.pi**3)/pc.hbar
	  # I will include the error on eta_EW via error propagation by hand
	  dN 		= pc.deta_EW*pc.GF**2./(24*np.pi**3)/pc.hbar
	  NA		= 1
	  val_l		= pt.particle()
	  val_l.samples  = np.r_['1',self.samples,np.array([self.VCKM[:,-1]]).T]
	  val 		= val_l.bs_val()
	  dval 		= val_l.bs_dval()
	  Cov		= val_l.covariance()
	  Cov0		= val_l.covariance()
	  Cov0[-1,:]	= 0
	  Cov0[:,-1]	= 0
	  Cov		= val_l.covariance()
	  cols 		= ['b','g']
	  for ii,(ml,sml) in enumerate(zip([pc.mtau,pc.mmu],['tau','mu'])):
	   l		= []
	   fig,ax 	= plt.subplots(1,1,figsize=(8,6))
	   # compute differential decay rate including Vub computed with own data
	   f		= lambda val: self.reduced_gamma_0_BGL(qsq,ml,val[:-1])*val[-1]**2
	   qsq 		= np.linspace(ml**2,self.tm,1000)
	   # first with our own exclusive CKM
	   dGdqsq 	= N*f(val)
	   ddGdqsq0	= np.sqrt((N*self.dfunc(f,val,Cov0,qsq))**2+(dN*dGdqsq)**2)
	   ddGdqsq	= np.sqrt((N*self.dfunc(f,val,Cov,qsq))**2+(dN*dGdqsq)**2)
	   ax.plot(qsq,dGdqsq,'r-' )
	   ax.fill_between(qsq,dGdqsq-ddGdqsq,dGdqsq+ddGdqsq,color='r' ,alpha=.4)
	   l.append(ax.fill_between(qsq,dGdqsq-ddGdqsq0,dGdqsq+ddGdqsq0,color='r' ,alpha=.2))

	   for iCKM,sVCKMinc in enumerate(['VCKM inclusive']):
	    VCKMinc	= self.experimental_input[sVCKMinc]['PDG 2021']['val']
	    dVCKMinc	= self.experimental_input[sVCKMinc]['PDG 2021']['dval']
	    f		= lambda val: self.reduced_gamma_0_BGL(qsq,ml,val[:-1])
	    dGdqsq 	= N*f(val) * VCKMinc**2
	    ddGdqsq0	= np.sqrt((N*self.dfunc(f,val,Cov0,qsq) * VCKMinc**2)**2+ (dN*dGdqsq*VCKMinc**2)**2)
	    ddGdqsq 	= np.sqrt( ddGdqsq0**2 + (2*dGdqsq/VCKMinc*dVCKMinc)**2)
	    ax.plot(qsq,dGdqsq,cols[iCKM]+'-' )
	    l.append(ax.fill_between(qsq,dGdqsq-ddGdqsq,dGdqsq+ddGdqsq,color=cols[iCKM] ,alpha=.2))
	    ax.fill_between(qsq,dGdqsq-ddGdqsq0,dGdqsq+ddGdqsq0,color=cols[iCKM] ,alpha=.1)

	   ax.set_xlim((0,self.tm))
	   ax.set_ylabel('$d\Gamma(B_s\\to K\\'+sml+'\\nu_\\'+sml+')/dq^2\,[{\\rm ps}^-1{\\rm GeV}^{-2}]$')
	   ax.set_xlabel('$q^2 [{\\rm GeV}^2]$ ')
	   ax.legend(l,['$|V_{ub}|\\rm exclusive (this work)$'.replace(" ","\,"),
	         	'$|V_{ub}|\\rm inclusive$'.replace(" ","\,")
	         	],prop={'family':'serif'}, frameon=False)
	   fig.savefig('Report/plots/dGdqsq_'+sml+'_%d_%d.pdf'%(self.Kp,self.K0))
	   fig.clf()
	   ###################################
	   # AFB plots
	   # using new Vub
	   lA		= []
	   figA,axA 	= plt.subplots(1,1,figsize=(8,6))
	   fA		= lambda val: self.AFBker_BGL(qsq,ml,val[:-1])*val[-1]**2
	   AFB		= N*fA(val)
	   dAFB		= np.sqrt((N*self.dfunc(fA,val,Cov ,qsq))**2+(dN*AFB)**2)
	   dAFB0		= np.sqrt((N*self.dfunc(fA,val,Cov0,qsq))**2+(dN*AFB)**2)
	   axA.plot(qsq,AFB,'r-' )
	   lA.append(axA.fill_between(qsq,AFB-dAFB,AFB+dAFB,color='r' ,alpha=.4))
	   axA.fill_between(qsq,AFB-dAFB0,AFB+dAFB0,color='r' ,alpha=.2)
	   # usinv other Vub
	   fA		= lambda val: self.AFBker_BGL(qsq,ml,val[:-1])
	   for iCKM,sVCKMinc in enumerate(['VCKM inclusive']):
	    VCKMinc	= self.experimental_input[sVCKMinc]['PDG 2021']['val']
	    dVCKMinc	= self.experimental_input[sVCKMinc]['PDG 2021']['dval']
	    AFB		= N*fA(val)*VCKMinc**2
	    dAFB0	= np.sqrt((N*self.dfunc(fA,val,Cov0,qsq)*VCKMinc**2)**2+(dN*AFB*VCKMinc**2)**2)
	    dAFB 	= np.sqrt((N*self.dfunc(fA,val,Cov,qsq)*VCKMinc**2)**2+(dN*AFB*VCKMinc**2)**2)
	    dAFB		= np.sqrt(dAFB**2 +(2*AFB/VCKMinc*dVCKMinc)**2)
	    axA.plot(qsq,AFB,cols[iCKM]+'-' )
	    lA.append(axA.fill_between(qsq,AFB-dAFB,AFB+dAFB,color=cols[iCKM] ,alpha=.2))
	    axA.fill_between(qsq,AFB-dAFB0,AFB+dAFB0,color=cols[iCKM] ,alpha=.1)

	   axA.set_xlim((0,self.tm))
	   axA.set_ylabel('$\\mathcal{A}^{\\'+sml+'}_{\\rm FB}\,[{\\rm ps}^{-1}{\\rm GeV}^{-2}]$')
	   axA.set_xlabel('$q^2 [{\\rm GeV}^2]$')
	   axA.legend(lA,['$|V_{ub}|\\rm exclusive (this work)$'.replace(" ","\,"),
	         	'$|V_{ub}|\\rm inclusive$'.replace(" ","\,")
	         	],prop={'family':'serif'}, frameon=False)
	   if sml=='mu':
	    axA.set_yscale('log')
	    axA.set_ylim((1e-9,1e-6))
	   figA.savefig('Report/plots/AFB_m'+sml+'_%d_%d.pdf'%(self.Kp,self.K0))

	   ###################################
	   # make polarisation asymmetry plot
	   lA		= []
	   figA,axA 	= plt.subplots(1,1,figsize=(8,6))
	   fA		= lambda val: self.Apolker_BGL(qsq,ml,val[:-1])*val[-1]**2
	   Cov0[-1,:]	= 0
	   Cov0[:,-1]	= 0
	   Apol		= NA*fA(val)
	   dApol		= NA*self.dfunc(fA,val,Cov0,qsq)
	   dApol0	= NA*self.dfunc(fA,val,Cov0,qsq)
	   axA.plot(qsq,Apol,'r-' )
	   axA.fill_between(qsq,Apol-dApol0,Apol+dApol0,color='r' ,alpha=.4)
	   lA.append(axA.fill_between(qsq,Apol-dApol,Apol+dApol,color='r' ,alpha=.3))
	   for iCKM,sVCKMinc in enumerate(['VCKM inclusive']):
	    VCKMinc	= self.experimental_input[sVCKMinc]['PDG 2021']['val']
	    dVCKMinc	= self.experimental_input[sVCKMinc]['PDG 2021']['dval']
	    fA		= lambda val: self.Apolker_BGL(qsq,ml,val[:-1])
	    Apol		= NA*fA(val)*VCKMinc**2
	    dApol	= np.sqrt((NA*self.dfunc(fA,val,Cov,qsq)*VCKMinc**2)**2+
	         		(2*Apol*dVCKMinc)**2)
	    axA.plot(qsq,Apol,cols[iCKM]+'-' )
	    lA.append(axA.fill_between(qsq,Apol-dApol,Apol+dApol,color=cols[iCKM] ,alpha=.2))
	    axA.fill_between(qsq,Apol-dApol0,Apol+dApol0,color=cols[iCKM] ,alpha=.1)


	   axA.set_xlim((0,self.tm))
	   axA.set_ylabel('$\\bar\\mathcal{ A}^{\\'+sml+'}_{\\rm pol}\,[{\\rm ps}^{-1}{\\rm GeV}^{-2}]$')
	   axA.set_xlabel('$q^2 [{\\rm GeV}^2]$')
	   axA.legend(lA,['$|V_{ub}|\\rm exclusive (this work)$'.replace(" ","\,"),
	         	'$|V_{ub}|\\rm inclusive$'.replace(" ","\,")
	         	],prop={'family':'serif'}, frameon=False)
	   if sml=='mu':
	    axA.set_yscale('log')
	    axA.set_ylim((1e-6,5e-5))
	   figA.savefig('Report/plots/Apol_m'+sml+'_%d_%d.pdf'%(self.Kp,self.K0))



	def make_pheno_tables(self):
	 #################################
	 # Tables for observables
	 #################################
	 self.report_observables = True
	 f=open('Report/tables/BFF_observables.txt','w')
	 self.print_table_header(f,'l@{\hspace{1mm}}llllllllll')
	 if 'experimental_input' in self.input_dict.keys():
	  f.write('$K_+$&$K_0$&'+
				'\multicolumn{1}{c}{$f(q^2=0)$}&'+
				'\multicolumn{1}{c}{$R_{B_s\\to K}^{\\rm impr}$}&'+
				'\multicolumn{1}{c}{$R_{B_s\\to K}$}&'+
				'\multicolumn{1}{c}{$\\frac{\Gamma^\\tau}{|V_{ub}|^2}\,[\\frac 1{\\rm ps}]$}&'+
				'\multicolumn{1}{c}{$\\frac{\Gamma^\\mu}{|V_{ub}|^2}\,[\\frac 1{\\rm ps}]$}&'+
				'\multicolumn{1}{c}{$V^{\\rm low}_{\\rm CKM}$}&'+
				'\multicolumn{1}{c}{$V^{\\rm high}_{\\rm CKM}$}&'+
				'\multicolumn{1}{c}{$V^{\\rm full}_{\\rm CKM}$}&' +
				'\\\\\n')
	 else:
	  f.write('$K_+$&$K_0$&'+
				'\multicolumn{1}{c}{$f(q^2=0)$}&'+
				'\multicolumn{1}{c}{$R_{B_s\\to K}^{\\rm impr}$}&'+
				'\multicolumn{1}{c}{$R_{B_s\\to K}$}&'+
				'\multicolumn{1}{c}{$\\frac{\Gamma^\\tau}{|V_{ub}|^2}\,[\\frac 1{\\rm ps}]$}&'+
				'\multicolumn{1}{c}{$\\frac{\Gamma^\\mu}{|V_{ub}|^2}\,[\\frac 1{\\rm ps}]$}&'+
				'\\\\\n')
	 f.write('\hline\n')
	 for cases in self.Kcases:
	  Kp	= cases[0]
	  K0	= cases[1]
	  rep 	= self.report['results_%d_%d'%(Kp,K0)]
	  R_our = self.disperr([np.mean(rep['ratio_BGL'])],[np.std(rep['ratio_BGL'])])[0]
	  Rs 	= self.disperr(np.mean(rep['ratio'],axis=0)	  ,np.std(rep['ratio'],axis=0))
	  R 	= Rs[0]
	  Gamma_tau = Rs[5]
	  Gamma_mu  = Rs[6]
	  f0 	= self.disperr([np.mean(rep['f(0)'])]	  ,[np.std(rep['f(0)'])])[0]
	  if 'experimental_input' in self.input_dict.keys():
	   VCKM = self.disperr(np.mean(rep['VCKM'],0),np.std(rep['VCKM'],0))
	   f.write('%d&%d&'%(Kp,K0)+'%s&%s&%s&%s&%s&%s&%s&%s\\\\\n'%(f0,R_our,R,Gamma_tau,Gamma_mu,VCKM[0],VCKM[1],VCKM[2]))
	  else:
	   f.write('%d&%d&'%(Kp,K0)+'%s&%s&%s&%s&%s\\\\\n'%(f0,R_our,R,Gamma_tau,Gamma_mu))
	 f.write('\hline\hline\\\\\n')
	 self.print_table_footer(f)
	 f.close()
	 f=open('Report/tables/BFF_observables2.txt','w')
	 self.print_table_header(f,'l@{\hspace{1mm}}llllllllll')
	 if 'experimental_input' in self.input_dict.keys():
	  f.write('$K_+$&$K_0$&'+
				'\multicolumn{1}{c}{$I[\mathcal{A}_{\\rm FB}^\\tau]\,[\\frac 1{\\rm ps}]$}&'+
				'\multicolumn{1}{c}{$I[\mathcal{A}_{\\rm FB}^\\mu]\,[\\frac 1{\\rm ps}]$}&'+
				'\multicolumn{1}{c}{$\mathcal{\\bar A}_{\\rm FB}^\\tau$}&'+
				'\multicolumn{1}{c}{$\mathcal{\\bar A}_{\\rm FB}^\\mu$}&'+
				'\multicolumn{1}{c}{$I[\mathcal{A}_{\\rm pol}^\\tau]\,[\\frac 1{\\rm ps}]$}&'+
				'\multicolumn{1}{c}{$I[\mathcal{A}_{\\rm pol}^\\mu]\,[\\frac 1{\\rm ps}]$}&'+
				'\multicolumn{1}{c}{$\mathcal{\\bar A}_{\\rm pol}^\\tau$}&'+
				'\multicolumn{1}{c}{$\mathcal{\\bar A}_{\\rm pol}^\\mu$}&'+
				'\\\\\n')
	 else:
	  f.write('$K_+$&$K_0$&'+'$q^2=0$&$R_{\\rm our}$&$R$&$\\bar\mathcal{A}_{\\rm FB}^\\tau$&$\\bar\mathcal{a}_{\\rm fb}^\\mu$' +'&\\\\\n')
	 f.write('\hline\n')
	 for cases in self.Kcases:
	  Kp	= cases[0]
	  K0	= cases[1]
	  rep 	= self.report['results_%d_%d'%(Kp,K0)]
	  Rs 	= self.disperr(np.mean(rep['ratio'],axis=0)	  ,np.std(rep['ratio'],axis=0))
	  IAFBtau	= Rs[3]
	  IAFBmu 	= Rs[4]
	  AFBtau	= Rs[1]
	  AFBmu 	= Rs[2]
	  IApoltau 	= Rs[9]
	  IApolmu  	= Rs[10]
	  Apoltau 	= Rs[7]
	  Apolmu  	= Rs[8]
	  f0 	= self.disperr([np.mean(rep['f(0)'])]	  ,[np.std(rep['f(0)'])])[0]
	  if 'experimental_input' in self.input_dict.keys():
	   VCKM = self.disperr(np.mean(rep['VCKM'],0),np.std(rep['VCKM'],0))
	   f.write('%d&%d&'%(Kp,K0)+'%s&%s&%s&%s&%s&%s&%s&%s\\\\\n'%(IAFBtau,IAFBmu,AFBtau,AFBmu,
								     IApoltau,IApolmu,Apoltau,Apolmu))
	  else:
	   f.write('%d&%d&'%(Kp,K0)+'%s&%s&%s&%s&%s\\\\\n'%(f0,R_our,R,AFBtau,AFBmu))
	 f.write('\hline\hline\\\\\n')
	 self.print_table_footer(f)
	 f.close()
	
	def make_pheno_report(self,tag=''):
	  """
	   Create a Latex report summarising all main results
	  """
	  self.make_tables()
	  if tag=='':
	   fn = 'BFF_pheno_report.tex'
	  else:
	   fn = 'BFF_pheno_report_'+tag+'.tex'
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
	  f.write('\\title{Bayesian form factor fit -- pheno report}')
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
	  f.write('\\verb|%-20s|: %s\\\\\n' % ('large_t',str(self.large_t)))
	  f.write('\\verb|%-20s|: %s\\\\\n' % ('large_t_ibase',str(self.large_t_ibase)))
	  f.write('\\verb|%-20s|: %s\\\\\n' % ('alpha',str(self.angle)))
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
	  f.write('\subsection{Observables}\n')
	  f.write('\\begin{center}\n')
	  f.write('\\tiny')
	  f.write('\input{tables//BFF_observables.txt}\n')
	  f.write('\input{tables//BFF_observables2.txt}\n')
	  f.write('\end{center}\n')
	  f.write('\subsection{Form-factor plots}\n')
	  for cases in self.Kcases:
	   f.write('\subsubsection{$(K_+,K_0)$=(%d,%d)}\n'%(cases[0],cases[1]))
	   f.write('\\begin{center}\n')
	   f.write('\includegraphics[width=10cm]{{plots/plot_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
	   f.write('\\end{center}\n')
	  if 'experimental_input' in self.input_dict.keys():
	   f.write('\subsection{Partial-decay-rate plots}\n')
	   for cases in self.Kcases:
	    f.write('\subsubsection{$(K_+,K_0)$=(%d,%d)}\n'%(cases[0],cases[1]))
	    f.write('\\begin{center}\n')
	    f.write('\includegraphics[width=10cm]{{plots/dGdqsq_mu_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
	    f.write('\includegraphics[width=10cm]{{plots/dGdqsq_tau_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
	    f.write('\includegraphics[width=10cm]{{plots/AFB_mmu_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
	    f.write('\includegraphics[width=10cm]{{plots/AFB_mtau_%d_%d}.pdf}\n'%(cases[0],cases[1])) 
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
	def compute_Vub_BstoK(self):
	  # get the experimental input from the input_dict
	  EI		= self.input_dict['experimental_input']['BstoK LHCb2012']['observables']['RBF']
	  BBstoDs	= EI['B(BstoDs)'][0]
	  dstat_BBstoDs	= EI['B(BstoDs)'][1]
	  dsyst_BBstoDs	= EI['B(BstoDs)'][2]
	  tauBs	 	= pc.tauBs
	  dtauBs 	= pc.dtauBs
	  self.VCKM 	= np.empty((self.N,0))
	  self.VCKM_zero= np.empty((self.N,0)) # _zero is result with zero exp error

	  # symmetrise errors
	  v		= EI['B(BstoDs)'] 
	  BBstoDs,dstat_BBstoDs,dsyst_BBstoDs = self.symmetrise(v[0],v[1],v[2],v[3],v[4])
	  v		= EI['RBF'][0]
	  RBF0,dstat_RBF0,dsyst_RBF0 = self.symmetrise(v[0],v[1],v[2],v[3],v[4])
	  v		= EI['RBF'][1]
	  RBF1,dstat_RBF1,dsyst_RBF1 = self.symmetrise(v[0],v[1],v[2],v[3],v[4])
	
	  # combine experimental result into data vector
	  v		= np.r_[RBF0,RBF1 ,BBstoDs      ,tauBs]
	  d_stat	= np.r_[dstat_RBF0,dstat_RBF1,dstat_BBstoDs,dtauBs]
	  d_syst	= np.r_[dsyst_RBF0,dsyst_RBF1,dsyst_BBstoDs,0]
	  cov_stat	= np.diag(d_stat**2)
	  # we assume systematic errors to be correlated 
	  cov_syst	= np.diag(d_syst)@np.array([[1,1,1,0],[1,1,1,0],[1,1,1,0],[0,0,0,1]])@np.diag(d_syst)
	  # generate samples for input following combined covariance matrix
	  samples 	= np.random.multivariate_normal(v,1*(cov_stat+cov_syst),size=self.N)
	  samples_zero 	= np.random.multivariate_normal(v,0*(cov_stat+cov_syst),size=self.N)
	  # combine lattice and experimental input
	  allsamples	= np.r_['1',self.samples,samples,self.eta_EW_samples[:,None]]
	  allsamples_zero= np.r_['1',self.samples,samples_zero,self.eta_EW_samples[:,None]]
	  res = pt.particle()
	  # compute Vub lower bin
	  qsqmin	= EI['qsq bins'][0][0]
	  qsqmax	= EI['qsq bins'][0][1]
	  dum		= np.delete(allsamples,self.Kp+self.K0+1,1)
	  res 		= my_apply_along_axis(self.fVCKM,dum,self.Kp,self.K0,qsqmin,qsqmax,parallel=False)
	  self.VCKM	= np.r_['1',self.VCKM,res]
	  dum		= np.delete(allsamples_zero,self.Kp+self.K0+1,1)
	  res 		= my_apply_along_axis(self.fVCKM,dum,self.Kp,self.K0,qsqmin,qsqmax,parallel=False)
	  self.VCKM_zero= np.r_['1',self.VCKM_zero,res]
	  # upper bin
	  qsqmin	= EI['qsq bins'][1][0]
	  qsqmax	= EI['qsq bins'][1][1]
	  dum		= np.delete(allsamples,self.Kp+self.K0+0,1)
	  res 		= my_apply_along_axis(self.fVCKM,dum,self.Kp,self.K0,qsqmin,qsqmax,parallel=False)
	  self.VCKM	= np.r_['1',self.VCKM,res]
	  dum		= np.delete(allsamples_zero,self.Kp+self.K0+0,1)
	  res 		= my_apply_along_axis(self.fVCKM,dum,self.Kp,self.K0,qsqmin,qsqmax,parallel=False)
	  self.VCKM_zero= np.r_['1',self.VCKM_zero,res]
	  print('VCKM laterr:   ',self.disperr(np.mean(self.VCKM_zero,0),np.std(self.VCKM_zero,0)))
	  print('VCKM toterr:   ',self.disperr(np.mean(self.VCKM,0),np.std(self.VCKM,0)))
	  # compute uncorrelated weighted average on each bin:
	  dat		= pt.particle()
	  dat.samples 	= self.VCKM_zero
	  cov		= dat.covariance()
	  invcov	= np.linalg.inv(cov)
	  val		= dat.bs_val()
	  dav_zero   	= 1./np.sum(invcov) 
	  av_zero    	= dav_zero*np.sum(invcov@dat.samples.T,0).T
	  print('VCKM_laterr:   ',self.disperr([np.mean(av_zero)],[np.sqrt(dav_zero)]))
	  self.VCKM_zero= np.append(self.VCKM_zero,np.array([av_zero]).T,1)
	  # compute uncorrelated weighted average on each bin:
	  dat		= pt.particle()
	  dat.samples 	= self.VCKM
	  cov		= dat.covariance()
	  (dhigh,dlow)	= dat.bs_dval()
	  Vhigh		= dat.samples[:,0]
	  Vlow 		= dat.samples[:,1]
	  av		= (Vlow/dlow**2+Vhigh/dhigh**2)/(1./dlow**2+1./dhigh**2)
	  dav		= np.std(av)
	  print('VCKM      :   ',self.disperr([np.mean(av)],[dav]))
	  self.VCKM	= np.append(self.VCKM,np.array([av]).T,1)
	##############################################################
	def make_observables(self,a):
	 """
	  This routine computes a comprehensive set of observables
	 """
	 self.ff.Kp	= self.Kp
	 self.ff.K0	= self.K0
	 # generate the MC for the prefactor eta_EW 
	 self.eta_EW_samples 	= np.random.normal(pc.eta_EW,pc.deta_EW,size=self.N)
	 # we feed this into the various routines by adding it as an additional column to the
	 # array with BGL-coefficient samples
	 a_eta_EW		= np.r_['1',a,self.eta_EW_samples[:,None]]
	 if 2==2:
	  print(self.separator)
	  print('BFF: making observables')
	  print(self.separator)
	  qsqmax 		= self.tm
	  # compute fp(0)
	  print(a.shape,self.Kp)
	  self.fp0		= my_apply_along_axis(self.ff.zfit_BGL_p_fn,a[:,:self.Kp],0,parallel=False)
	  # compute f0(0)
	  self.f00		= my_apply_along_axis(self.ff.zfit_BGL_0_fn,a[:,self.Kp:],0,parallel=False)
	  # compute fp(qsqmax)
	  self.fpqsqmax		= my_apply_along_axis(self.ff.zfit_BGL_p_fn,a[:,:self.Kp],self.tm,parallel=False)
	  # compute f0(qsqmax)
	  self.f0qsqmax		= my_apply_along_axis(self.ff.zfit_BGL_0_fn,a[:,self.Kp:],self.tm,parallel=False)
	  print('qsq0_p_BGL:   ',self.disperr([np.mean(self.fp0)],[np.std(self.fp0)]))
	  print('qsq0_0_BGL:   ',self.disperr([np.mean(self.f00)],[np.std(self.f00)]))
	  print('qsqmax_p_BGL: ',self.disperr([np.mean(self.fpqsqmax)],[np.std(self.fpqsqmax)]))
	  print('qsqmax_0_BGL: ',self.disperr([np.mean(self.f0qsqmax)],[np.std(self.f0qsqmax)]))
	  # compute improved R ratio with \ell = m_muon
	  self.ratios_BGL	= my_apply_along_axis(self.improved_R   ,a,pc.mmu,pc.mtau**2,qsqmax,parallel=False)
	  print('ratios_BGL:   ',self.disperr([np.mean(self.ratios_BGL)],[np.std(self.ratios_BGL)]))
	  # compute improved R ratio with \ell = m_electron
	  self.ratios_BGLe	= my_apply_along_axis(self.improved_R   ,a,0.511,pc.mtau**2,qsqmax,parallel=False)
	  print('ratios_BGLe:   ',self.disperr([np.mean(self.ratios_BGL)],[np.std(self.ratios_BGL)]))
	  #  compute the asymetries
	  self.ratio 		= my_apply_along_axis(self.gen_R_AFB_BGL,a_eta_EW,pc.mtau,pc.mmu,parallel=False)
	  print('ratios     :  ',self.disperr([np.mean(self.ratio[:,0])],[np.std(self.ratio[:,0])]))
	  print('AFB tau    :  ',self.disperr([np.mean(self.ratio[:,1])],[np.std(self.ratio[:,1])]))
	  print('AFB mu     :  ',self.disperr([np.mean(self.ratio[:,2])],[np.std(self.ratio[:,2])]))
	  print('I AFB tau  :  ',self.disperr([np.mean(self.ratio[:,3])],[np.std(self.ratio[:,3])]))
	  print('I AFB mu   :  ',self.disperr([np.mean(self.ratio[:,4])],[np.std(self.ratio[:,4])]))
	  print('intG tau   :  ',self.disperr([np.mean(self.ratio[:,5])],[np.std(self.ratio[:,5])]))
	  print('intG mu    :  ',self.disperr([np.mean(self.ratio[:,6])],[np.std(self.ratio[:,6])]))
	  print('barApol tau:  ',self.disperr([np.mean(self.ratio[:,7])],[np.std(self.ratio[:,7])]))
	  print('barApol mu :  ',self.disperr([np.mean(self.ratio[:,8])],[np.std(self.ratio[:,8])]))
	  print('Apol tau   :  ',self.disperr([np.mean(self.ratio[:,9])],[np.std(self.ratio[:,9])]))
	  print('Apol mu    :  ',self.disperr([np.mean(self.ratio[:,10])],[np.std(self.ratio[:,10])]))

	 # compute CKM ME	
	 if 'experimental_input' in self.input_dict.keys():
	  self.VCKM = []
	  if self.input_dict['decay']=='BstoK':
	   self.compute_Vub_BstoK()
	  else:
	   print('No VCKM computation implemented for decay channel %s'%self.input_dict['decay'])
	 self.report['results_%d_%d'%(self.Kp,self.K0)] = {**self.report['results_%d_%d'%(self.Kp,self.K0)] , **{
	      'ratio':	self.ratio,
	      'ratio_BGL':	self.ratios_BGL,
	      'f(0)':		self.fp0,
	      'fp(qsqmax)':	self.fpqsqmax,
	      'f0(qsqmax)':	self.f0qsqmax,
	      'AFB tau'    : self.ratio[:,1]   ,
              'AFB mu'     : self.ratio[:,2],
              'I AFB tau'  : self.ratio[:,3],
              'I AFB mu'   : self.ratio[:,4],
              'intG tau'   : self.ratio[:,5],
              'intG mu'    : self.ratio[:,6],
              'barApol tau': self.ratio[:,7],
              'barApol mu' : self.ratio[:,8],
              'Apol tau'   : self.ratio[:,9],
              'Apol mu'    : self.ratio[:,10]
	      }}

	 if 'experimental_input' in self.input_dict.keys():
	  self.report['results_%d_%d'%(self.Kp,self.K0)]['VCKM'] 	= self.VCKM
	  self.report['results_%d_%d'%(self.Kp,self.K0)]['VCKM_zero'] 	= self.VCKM_zero
