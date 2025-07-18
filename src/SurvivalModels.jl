module SurvivalModels

# Write your package code here.
using Optimization, LinearAlgebra, Optim, ForwardDiff, StatsBase, Random, Distributions, DataFrames, StatsModels, SurvivalBase, StatsAPI, OptimizationOptimJL, SurvivalDistributions

using StatsBase: fit
using SurvivalBase: Surv, Strata

include("NonParametric/KaplanMeier.jl")
include("NonParametric/LogRankTest.jl")
include("Semiparametric/Cox.jl")
include("Parametric/GeneralHazard.jl")

export fit, KaplanMeier, LogRankTest, Cox, @formula, Surv, Strata, confint, GeneralHazard, ProportionalHazard, AcceleratedFaillureTime, AcceleratedHazard, simGH

end
