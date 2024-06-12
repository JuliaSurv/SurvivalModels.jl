module SurvivalModels

# Write your package code here.

using SurvivalBase: Surv, Strata
using StatsAPI
using StatsBase
using StatsModels
using DataFrames

include("NonParametric/KaplanMeier.jl")
include("NonParametric/LogRankTest.jl")

export KaplanMeier, LogRankTest
end
