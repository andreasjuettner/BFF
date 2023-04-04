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
 
## Execution
The following example script reproduces the results presented in the paper. Upon execution 
Bayesian inference is run for various combinations of lattice and sum-rule input and for
a number of truncations of the BGL expansion. For each set of input data a LaTeX report
is generated.

To run it:

```python run_BstoK.py```
