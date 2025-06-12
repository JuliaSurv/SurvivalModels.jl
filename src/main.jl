### This file is supposed to produce the benchmarks between the several versions of Cox that we have, on various datasets with various features. 

### It should not do anything else, and no other julia file than main.jl and tests.jl should be main entry points. Launching these two files should be enough to reproduce all the results. 


using RDatasets, Optimization, LinearAlgebra, Optim, OptimizationOptimJL, 
      BenchmarkTools, Test, RCall, PyCall, ForwardDiff, Survival, StatsBase,
      Plots, CSV, Random, Distributions, DataFrames, Conda, TestItemRunner, StatsModels

# Include the code: 
includet("Semiparametric/Cox/Cox.jl")
#includet("Semiparametric/Cox/vR.jl")
includet("Semiparametric/Cox/vJ.jl")
includet("Semiparametric/Cox/v1.jl")
includet("Semiparametric/Cox/v2.jl") 
includet("Semiparametric/Cox/v3.jl")
includet("Semiparametric/Cox/v4.jl")
includet("Semiparametric/Cox/v5.jl")

