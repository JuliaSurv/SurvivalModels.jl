module SurvivalModels

# Write your package code here.
using ADTypes, LinearAlgebra, Optim, ForwardDiff, StatsBase, Random, Distributions, DataFrames, StatsModels, SurvivalBase, StatsAPI, SurvivalDistributions

using StatsBase: fit, predict
using SurvivalBase: Surv, Strata

include("utils.jl")
include("NonParametric/KaplanMeier.jl")
include("NonParametric/LogRankTest.jl")
include("Semiparametric/Cox.jl")
include("Parametric/GeneralHazard.jl")
include("Metrics/BrierScore.jl")

export fit, predict, KaplanMeier, LogRankTest, Cox, @formula, Surv, Strata, confint, GeneralHazard, ProportionalHazard, AcceleratedFaillureTime, AcceleratedHazard, PHMethod, AFTMethod, AHMethod, GHMethod, simulate, simGH, predict_survival, predict_expected, loss, loglikelihood, aic, aicc, bic, dof, nobs, coef, vcov, stderror

end
