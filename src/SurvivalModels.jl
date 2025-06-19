module SurvivalModels

# Write your package code here.
using Optimization, LinearAlgebra, Optim, ForwardDiff, Survival, StatsBase, Random, Distributions, DataFrames, StatsModels, SurvivalBase, StatsAPI, OptimizationOptimJL

using StatsBase: fit

include("NonParametric/KaplanMeier.jl")
include("NonParametric/LogRankTest.jl")

include("Semiparametric/Cox/Cox.jl")
include("Semiparametric/Cox/v0.jl")
include("Semiparametric/Cox/v1.jl")
include("Semiparametric/Cox/v2.jl")
include("Semiparametric/Cox/v3.jl")
include("Semiparametric/Cox/v4.jl")
include("Semiparametric/Cox/v5.jl")
include("Semiparametric/Cox/vJ.jl")

export fit, KaplanMeier, LogRankTest, Cox, @formula, Surv
end
