# SelectionBias.jl
This repo contains the code to reproduce the experiments from the paper 'Learning From Missing Data Using Selection Bias in Movie Recommendation' (https://arxiv.org/abs/1509.09130) 

# Simulations
 
The simulation folder contains a python library that allows to reproduce the figures from Section 4. The my_optimisation.py file contains the likelihood, objective and gradient functions that are used to solve the optimisation problem (3) in Section 3.1. The scripts simulations.py and streaming_simu.py use these functions to estimate the missing entries of the matrix.

# Real Data

This set of function uses the library IncrementalSVD.jl in order to compute a low rank approximation of the movie lens datasets. This library has a function load_large_movielens_dataset() that automatically imports the movielens10M to compute its SVD. Code is written to be run in parallel, using for instance

julia _p 4 test.jl

to run the code on 4 independent units. 
