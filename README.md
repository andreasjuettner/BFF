# BFF -- Bayesian Form factor Fit

## Author:
 - Andreas Juettner    <andreas.juttner@cern.ch>

## Bibliographic reference for code:
[![DOI](https://zenodo.org/badge/623083004.svg)](https://zenodo.org/badge/latestdoi/623083004)

## Requirements:

```Python 3```, ```numpy```, ```h5py```, ```os```, ```scipy```, ```matplotlib```, ```getdist```, ```gitpython```

## Introduction:
This library implements the novel method for fitting form factors presented in the paper
[Bayesian inference for form-factor fits regulated by unitarity and analyticity](https://arxiv.org/abs/2303.11285)
by J.M. Flynn, A. Jüttner and J.T. Tsang. 

Abstract: We propose a model-independent framework for fitting hadronic form-factor data, which is often only available at discrete kinematical points, using parameterisation based on to unitarity and analyticity. In this novel approach the latter two properties of quantum-field theory regulate the ill-posed fitting problem and allow model-independent predictions over the entire physical range. Kinematical constraints, for example for the vector and scalar form factors in semileptonic meson decays, can be imposed exactly. The  core formulae are straight-forward to implement with standard math libraries. We take account of a generalisation of the original Boyd~Grinstein~Lebed (BGL) unitarity constraint for form factors and demonstrate our method for the exclusive semileptonic decay $B_s\to K \ell \nu$, for which we make a number of phenomenologically relevant predictions, including  the CKM matrix element $|V_{ub}|$.
 
## Installation
The BFF library depends on the submodule particles (helper class for resampling).

In order to correctly install BFF including the submodule please run

git clone --recurse-submodules https://github.com/andreasjuettner/BFF.git

## Execution
The following example script reproduces the results presented in the paper. Upon execution 
Bayesian inference is run for various combinations of lattice and sum-rule input and for
a number of truncations of the BGL expansion. For each set of input data a LaTeX report
is generated.

To run it:

```python run_BstoK.py```

## Specifying BGL fit function
The target BGL fit function needs to be specified in the run script. For the example of a $B_s\to K\ell\nu$ fit
in `run_BstoK.py` this is:
```
input_dict = {
'decay':       'BstoK',
'Mi':          # initial-state mass in GeV (float)
'Mo':          # final-state mass in GeV (float)
'sigma':       # sigma for prior in algorithm  (float)
'Kp':          # target Kp (BGL truncation) - can be changed later (int)
'K0':          # target K0 (BGL truncation) - can be changed later (int)
'tstar':       '29.349570696829012', # string, will be evaluated with Python's eval
't0':          'self.tstar - np.sqrt(self.tstar\*(self.tstar-self.tm))', 
                                     # string, will be evaluated with Python's eval
'chip':        # susceptibility fp (float)
'chi0':        # susceptibility f0 (float)
'mpolep':      [pole1,pole2,...], # list of pole positions in GeV
'mpole0':      [],                # list of pole positions in GeV
'N'    :       # number of desired samples (int)
'outer_p':     [nI,K,a,b], # specs for outer function fp, K is string, e.g. '48\*np.pi'
'outer_0':     [nI,K,a,b], # specs for outer function f0, K is string 
'seed':        # RNG seed
'experimental_input': experimental_input
}
```
The format `[nI,K,a,b]` for `outer_p` and `outer_0` becomes clear from the definition of the outer function in `lib/zfit_lib.py` as 
```
def outer_phi_ker(self,qsq,a,b,nI,K,chi):
 rq     = np.sqrt(self.tstar - qsq)
 r0     = np.sqrt(self.tstar - self.t0)
 rm     = np.sqrt(self.tstar - self.tm)
 res = np.sqrt(nI/K/chi)\
        * np.sqrt(rq)/np.sqrt(r0)\
        * (rq           + r0                    )\
        * (rq           + np.sqrt(self.tstar)   )**(-(b+3))\
        * (self.tp      - qsq                   )**(a*1./4)\
        * (rq           + rm                    )**(a*1./2)
 return res
```
## Input format:
All input data is provided in terms of a Python dictionary, one dictionary entry per data set.  Currently the input can be provided in terms of form-factor results at reference $q^2$ values with covariance matrix (`'data type':'ff'`), or in terms of BCL parameters with covariance matrix (`'data type':'BCL'`). The format is as follows:

- form-factor values available at reference $q^2$ points -- example RBC/UKQCD 23:
```
'RBCUKQCD 23 lat':
{
'data type':'ff', # type of input
'label':          # label for plots etc. (string)
'Np':             # number of data points f+ (int)
'N0':             # number of data points f0 (int)
'qsqp':	          # f+ qsq reference points (numpy array)
'qsq0':	          # f0 qsq reference points (numpy array)
'fp':             # f+ values (numpy array)
'f0':             # f0 values`(numpy array)
Cff':             # {f+,f0} covariance matrix (numpy array)
}
```

- Input in terms of BCL fit parameters -- example HPQCD 14. The code creates synthetic reference $q^2$ values for further processing in the fit.
```
'HPQCD 14':
{    
 'journal': ' Phys.Rev.D 90 (2014) 054506',       # just for internal reference
 'data type': 'BCL',                              # data type BCL    
 'label':     # internal label for plots (string)
 'Kp':        # BCL order for f+ (int)
 'K0':        # BCL order for f0 (int)
 'polep':     # pole for f+ (list of floats)
 'pole0':     # pole for f0 (list of floats)
 'tstar':     # threshold, string, will be evaluated with eval
 't0':        # BCL parameter t0, string, will be evaluated with eval
 'tm':        # BCL parameter tm, string, will be evaluated with eval
 'tp':        # BCL parameter tp, string, will be evaluated with eval
 'qsqp':      # synthetic data for f+ will be generated for these qsq values (numpy array)
 'qsq0':      # synthetic data for f0 will be generated for these qsq values (numpy array)
 'bp':        # BCL parameters for f+ (numpy array)
 'b0':        # BCL parameters for f0 (numpy array)
 'Cp0':       # covariance matrix for BCL (numpy array)
}
```
