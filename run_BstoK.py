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
##############################################################################
#import lib.BFF_lib as Bff
import lib.BFF_pheno_lib as Bff
import lib.physical_constants as pc
import lib.data_BstoK as data
import numpy as np
import h5py
import os
import os.path


N=2000	 # number of Bayesian inference samples


separator = '##########################################'
######################################################################################
# experimental inpute required for VCKM-determination
######################################################################################
experimental_input = {
	'BstoK LHCb2012':{
	'joural'	:'PHYSICAL REVIEW LETTERS 126, 081804 (2021)',
	'observables'	:{
		'RBF'	:{ 'qsq bins':	[[pc.mmu**2,7],[7,(pc.mBsphys-pc.mKphys)**2],[pc.mmu**2,(pc.mBsphys-pc.mKphys)**2]],
			   'RBF'	:	1e-3*np.array(
				[[1.66,0.08,0.08,np.sqrt(0.07**2+0.05**2)	,np.sqrt(0.07**2+0.05**2)],
				 [3.25,0.21,0.21,np.sqrt(0.16**2+0.09**2)	,np.sqrt(0.17**2+0.09**2)],
				 [4.89,0.21,0.21,np.sqrt(0.20**2+0.14**2)	,np.sqrt(0.21**2+0.14**2)]]),
			   'B(BstoDs)':	1e-2*np.array([ 2.49,0.12,0.12,np.sqrt(+.14**2+.16**2)	,np.sqrt(+0.14**2+0.16**2)]) },
			}
		},
	# inclusive Vub for comparison 
	'VCKM inclusive': {
			'PDG 2021': { 	'val' :4.13e-3,
					'dval':0.26e-3}
			}
	}
######################################################################################
# specify input for BGL fit
######################################################################################
input_dict = {
	 'decay':	'BstoK',
         'Mi':		pc.mBsphys,	# initial-state mass
         'Mo':		pc.mKphys,	# final-state mass 
         'sigma':	.5,		# sigma for prior in algorithm
	 'Kp':		4,		# target Kp (BGL truncation) - can be changed later
	 'K0':		4,		# target K0 (BGL truncation) - can be changed later
	 'tstar':	'29.349570696829012', # value of t*
	 't0':		'self.tstar - np.sqrt(self.tstar*(self.tstar-self.tm))', # definition of t0
	 'chip':	pc.chip_BstoK,	# susceptibility fp	
	 'chi0':	pc.chi0_BstoK,  # susceptibility f0
	 'mpolep':	[pc.mBstar],	# fplus pole
	 'mpole0':	[],		# fzero pole (no pole for BstoK)
	 'N'	:	N,		# number of desired samples
         'outer_p':	[1,'48*np.pi',3,2], # specs for outer function fp
         'outer_0':	[1,'16*np.pi/(self.tp*self.tm)',1,1], # specs for outer function f0
	 'seed':	123,		# RNG seed
	 'experimental_input': experimental_input
	}


######################################################################################
# PREPARE form-factor data
######################################################################################
# Read the data from the RBCUKQCD 23 BstoK paper
f=h5py.File('data/BstoK_paper_data.h5','r')

# prepare FNAL/MILC covariance from errors and correlation
dval		= data.data['FNALMILC 19']['dval']
corr		= data.data['FNALMILC 19']['corr']
FNALMILC_cov 	= np.dot(np.diag(dval),np.dot(corr,np.diag(dval)))

# prepare RBC/UKQCD 23 data:
Np_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/Np'))
N0_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/N0'))
ff_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/ff'))
corr_stat_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/corr_stat'))
corr_syst_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/corr_syst'))
corr_flat_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/corr_flat'))
d_stat_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/d_stat'))
d_syst_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/d_syst'))
d_flat_RBCUKQCD23	= np.array(f.get('RBCUKQCD23_lat/ref_ff/d_flat'))
C_RBCUKQCD23		=(np.diag(d_stat_RBCUKQCD23)@corr_stat_RBCUKQCD23@np.diag(d_stat_RBCUKQCD23)
			 +np.diag(d_syst_RBCUKQCD23)@corr_syst_RBCUKQCD23@np.diag(d_syst_RBCUKQCD23)
			 +np.diag(d_flat_RBCUKQCD23)@corr_flat_RBCUKQCD23@np.diag(d_flat_RBCUKQCD23))
# now set up input dict
input_data = {
	'RBCUKQCD 23 lat':
	{
 	 'data type':	'ff',
	 'label'	:'RBC/UKQCD 23',
         'Np':		Np_RBCUKQCD23,
         'N0':		N0_RBCUKQCD23,
         'qsqp':	pc.qsqp_ref_BstoK[[0,2]],
         'qsq0':	pc.qsq0_ref_BstoK,
	 'fp':		ff_RBCUKQCD23[:Np_RBCUKQCD23],
	 'f0':		ff_RBCUKQCD23[Np_RBCUKQCD23:],
	 'Cff':		C_RBCUKQCD23,
	},
	'HPQCD 14':
	{	
	 'journal': 'Phys.Rev.D 90 (2014) 054506',
 	 'data type':	'BCL',
	 'label':	'HPQCD 14',
         'Kp':		data.data['HPQCD 14']['Kp'],
         'K0':		data.data['HPQCD 14']['K0'],
	 'polep':	[5.32520],
	 'pole0':	[5.6794],
	 'tstar':	'(pc.mBsphys+pc.mKphys)**2',#'self.tp',
	 't0':		'(pc.mBsphys+pc.mKphys)*(np.sqrt(pc.mBsphys)-np.sqrt(pc.mKphys))**2',
	 'tm':		'(pc.mBsphys-pc.mKphys)**2',
	 'tp':		'(pc.mBsphys+pc.mKphys)**2',
         'qsqp':	np.linspace(17,(pc.mBsphys-pc.mKphys)**2,3),#pc.qsqp_ref_BstoK,
         'qsq0':	np.linspace(17,(pc.mBsphys-pc.mKphys)**2,3),#pc.qsq0_ref_BstoK,
	 'bp':		data.data['HPQCD 14']['val'][:data.data['HPQCD 14']['Kp']],
	 'b0':		data.data['HPQCD 14']['val'][data.data['HPQCD 14']['Kp']:],
	 'Cp0':		data.data['HPQCD 14']['cov']
	},
	'FNALMILC 19':
	{	
	 'journal': 'PHYS. REV. D 100, 034501 (2019)',
 	 'data type':	'BCL',
	 'label':	'FNAL\/MILC 19',
         'Kp':		data.data['FNALMILC 19']['Kp'],
         'K0':		data.data['FNALMILC 19']['K0'],
	 'polep':	[5.32465],
	 'pole0':	[5.68],
	 'tstar':	'(pc.mBphys+pc.mpiphys)**2',#'self.tp',
	 't0':		'self.tstar - np.sqrt(self.tstar*(self.tstar-self.tm))',
	 'tm':		'(pc.mBsphys-pc.mKphys)**2',
	 'tp':		'(pc.mBsphys+pc.mKphys)**2',
         'qsqp':	np.linspace(17,(pc.mBsphys-pc.mKphys)**2,3),#pc.qsqp_ref_BstoK,
         'qsq0':	np.linspace(17,(pc.mBsphys-pc.mKphys)**2,3),#pc.qsq0_ref_BstoK,
	 'bp':		data.data['FNALMILC 19']['val'][:data.data['FNALMILC 19']['Kp']],
	 'b0':		data.data['FNALMILC 19']['val'][data.data['FNALMILC 19']['Kp']:],
	 'Cp0':		FNALMILC_cov
	},
	'Khodjamirian 17': # sum-rule data
	{
	 'journal'	:'JHEP08(2017)112',
	 'label'	: 'Khodjamirian 17',
	 'data type'	: 'ff',
         'Np':		1,
         'N0':		0,
         'qsqp':	np.array([0]),
         'qsq0':	np.array([]),
	 'fp':		np.array([0.336]),
	 'f0':		np.array([]),
	 'Cff':		np.array([[0.023**2]])
	},
	'Duplancic 08': # sum-rule data
	{
	 'journal'	:'PRD 78 054015 (2008)',
	 'label'	: 'Duplancic 08',
	 'data type'	: 'ff',
         'Np':		1,
         'N0':		0,
         'qsqp':	np.array([0]),
         'qsq0':	np.array([]),
	 'fp':		np.array([0.305]),
	 'f0':		np.array([]),
	 'Cff':		np.array([[0.035**2]])
	},
	'Faustov 13': # sum-rule data
	{
	 'journal'	:'PRD 87 094028 (2013)',
	 'label'	: 'Faustov 13',
	 'data type'	: 'ff',
         'Np':		1,
         'N0':		0,
         'qsqp':	np.array([0]),
         'qsq0':	np.array([]),
	 'fp':		np.array([0.284]),
	 'f0':		np.array([]),
	 'Cff':		np.array([[0.014**2]])
	},
	'Wang 12': # sum-rule data
	{
	 'journal'	:'PRD 86 114025 (2012)',
	 'label'	: 'Wang 12',
	 'data type'	: 'ff',
         'Np':		1,
         'N0':		0,
         'qsqp':	np.array([0]),
         'qsq0':	np.array([]),
	 'fp':		np.array([0.265]),
	 'f0':		np.array([]),
	 'Cff':		np.array([[0.035**2]])
	}
}
f.close()
######################################################################################
######################################################################################
# Pick and choose which data sets should be analysed. Will be looped over
variations 	=  [    ['RBCUKQCD 23 lat'],
			['RBCUKQCD 23 lat','HPQCD 14','Khodjamirian 17','Duplancic 08','Faustov 13','Wang 12'],
			['RBCUKQCD 23 lat','HPQCD 14','Khodjamirian 17'],	
			['FNALMILC 19'],
			['HPQCD 14'],
			['RBCUKQCD 23 lat','HPQCD 14'],
			['RBCUKQCD 23 lat','HPQCD 14','FNALMILC 19']]
variations 	=  [    ['RBCUKQCD 23 lat']]
# choose which (Kp,K0) variations you wish to evaluate. The format is
# [Kp,K0,sigma], where sigma may have to be tuned for good
# Monte-Carlo acceptance rate
Klist		= [	[2,2,0.52,1.],
		   	[2,3,0.52,1.],
			[3,2,0.52,1.],
			[3,3,0.52,1.],
			[3,4,0.52,1.],
		   	[4,3,.52,1.],
			[4,4,.52,1.], 
			[5,5,.5,1.0],
			[6,6,.5,1.5]]
#			[7,7,.65,1.0],
#			[8,8,.6,1.0],
#			[9,9,.42,1.],
#			[10,10,.32,1.]]
# Let's get started
for combi in variations:

 # instantiate BFF library
 bff 		= Bff.Bff_pheno(input_dict,large_t=False,large_t_ibase = 1)
 bff.add_HMChPT	= 0 # we don't want to plot overlay HMChPT results

 # now add all input for current variation 'combi'
 for c in combi:
  bff.add_data(input_data,c)

 print(separator)
 print('# Now doing ',' '.join(combi))

 # now loop through the list of desired (Kp,K0) cases

 for K in Klist:
  bff.Kp	= K[0]
  bff.K0	= K[1]
  bff.sigma 	= K[2]

  # reset seed such that every run starts with the same seed
  bff.rnd.seed(bff.seed)

  # do the job
  samples 	= bff.run_BFF()
  # generate plots
  bff.make_observables(bff.samples)
  bff.make_plots()
  bff.make_pheno_plots()
  bff.make_pheno_tables()

 # After having looped through all (Kp,K0) cases all results are still
 # stored in memory. We can now proceed to write the executive report:
 bff.make_report()

 # delete class
 del bff

 # create name for report 
 sbffl = 'and'.join(combi).replace(' ','_')
 # Create a directory where to store the report and h5dump of all results
 if not os.path.exists('Report_'+sbffl):
  os.system('mkdir Report_'+sbffl)
  print('Created directory Report_'+sbffl)

 # also create a library of all pdf reports
 if not os.path.exists('Reports'):
  os.system('mkdir Reports')
  print('Created directory Reports')
 os.system('cp Report/BFF_report.pdf Reports/BFF_report_'+sbffl+'.pdf')
 print('Copy of pdf report put into Reports/')
 os.system('cp -r Report/* Report_'+sbffl)
 print('Copy of Report directory created in Report_'+sbffl)

# Done

