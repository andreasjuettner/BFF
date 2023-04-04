# BFF -- Bayesian Form factor Fit

## Author:
 - Andreas Juettner    <andreas.juttner@cern.ch>

## Requirements:

```Python 3```, ```numpy```, ```h5py```, ```os```, ```scipy```, ```matplotlib```, ```getdist```, ```gitpython```

## Introduction:
This library implements the novel method for fitting form factors presented in the paper
*Bayesian inference for form-factor fits regulated by unitarity and analyticity*
by J.M. Flynn, A. Jüttner and J.T. Tsang. 

Abstract: We propose a model-independent framework for fitting hadronic form-factor
data, which is often only available at discrete kinematical points, to unitarity- and analyticity-
based parameterisations. In this novel approach the latter two properties of quantum-field
theory regulate this ill-posed problem and allow to determine model-independent predictions 
over the entire physical range. Kinematical constraints, which exist for example for
the vector and scalar form factor of semileptonic meson decays can be imposed exactly.
The core formulae are straight-forward to implement with standard math libraries and we
provide one such implementation as supplementary material. As well as proposing a generalised 
version of the Boyd Grinstein Lebed (BGL) unitarity constraint for form factors, we
demonstrate the novel method for the case of exclusive semileptonic decay $B_s\to K\ell\nu$, for
which we make a number of phenomenologically relevant predictions, such as for instance
the CKM matrix element $|V_{ub}|$.

## Execution
The following example script reproduces the results presented in the paper. Upon execution 
Bayesian inference is run for various combinations of lattice and sum-rule input and for
a number of truncations of the BGL expansion. For each set of input data a LaTeX report
is generated.

To run it:

```python run_BstoK.py```
