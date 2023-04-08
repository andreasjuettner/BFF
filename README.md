# BFF -- Bayesian Form factor Fit

## Author:
 - Andreas Juettner    <andreas.juttner@cern.ch>

## Requirements:

```Python 3```, ```numpy```, ```h5py```, ```os```, ```scipy```, ```matplotlib```, ```getdist```, ```gitpython```

## Introduction:
This library implements the novel method for fitting form factors presented in the paper
[Bayesian inference for form-factor fits regulated by unitarity and analyticity](https://arxiv.org/abs/2303.11285)
by J.M. Flynn, A. Jüttner and J.T. Tsang. 

Abstract: We propose a model-independent framework for fitting hadronic form-factor data, which is often only available at discrete kinematical points, using parameterisation based on to unitarity and analyticity. In this novel approach the latter two properties of quantum-field theory regulate the ill-posed fitting problem and allow model-independent predictions over the entire physical range. Kinematical constraints, for example for the vector and scalar form factors in semileptonic meson decays, can be imposed exactly. The  core formulae are straight-forward to implement with standard math libraries. We take account of a generalisation of the original Boyd~Grinstein~Lebed (BGL) unitarity constraint for form factors and demonstrate our method for the exclusive semileptonic decay $B_s\to K \ell \nu$, for which we make a number of phenomenologically relevant predictions, including  the CKM matrix element $|V_{ub}|$.
 
## Execution
The following example script reproduces the results presented in the paper. Upon execution 
Bayesian inference is run for various combinations of lattice and sum-rule input and for
a number of truncations of the BGL expansion. For each set of input data a LaTeX report
is generated.

To run it:

```python run_BstoK.py```

## Specifying BGL fit function
The target BGL fit function needs to be specified in the run script. For the example of a $B_s\to K\ell\nu$ fit
the example in  `run_BstoK.py` is
```
input_dict = {
         'decay':       'BstoK',
         'Mi':          pc.mBsphys,     # initial-state mass
         'Mo':          pc.mKphys,      # final-state mass
         'sigma':       .5,             # sigma for prior in algorithm
         'Kp':          4,              # target Kp (BGL truncation) - can be changed later
         'K0':          4,              # target K0 (BGL truncation) - can be changed later
         'tstar':       '29.349570696829012', # value of t*
         't0':          'self.tstar - np.sqrt(self.tstar*(self.tstar-self.tm))', # definition of t0
         'chip':        pc.chip_BstoK,  # susceptibility fp
         'chi0':        pc.chi0_BstoK,  # susceptibility f0
         'mpolep':      [pc.mBstar],    # fplus pole
         'mpole0':      [],             # fzero pole (no pole for BstoK)
         'N'    :       N,              # number of desired samples
         'outer_p':     [1,'48*np.pi',3,2], # specs for outer function fp
         'outer_0':     [1,'16*np.pi/(self.tp*self.tm)',1,1], # specs for outer function f0
         'seed':        123,            # RNG seed
         'experimental_input': experimental_input
        }
```

Most of the item in this dictionary should be self-explanatory:
```
Mi		mass of decaying meson in GeV
Mo		mass of produced meson in GeV
sigma		this is the width of the prior required in the algorith, see Sec. 3.2.2 of the paper
Kp/K0		these are the truncations of the vector and scalar form factor, resepctively
tstar 		Bpi threshold
t0		BGL parameter, Eq. (2.5)
chip/chi0	susceptibilities for vector and scalar channel
mpolep/mpole0	pole masses for the Blaschke factors
N		target number of samples
outer_p/outer_0	[\eta_I,J,K,a,b] -- see following code block for details
seed		seed for the random number generator
```
The outer function is defined in `lib/zfit_lib.py` as 
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
