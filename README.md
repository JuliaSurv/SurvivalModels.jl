# SurvivalModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/dev/)
[![Build Status](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.html)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# Roadmap

This package is for the moment empty. The goal of this implementation is to provide an iterface to fit on standard survival problems (that is, one censored time to event is observed, eventually with covariates), the following models: 

- Non-parametric: Kaplan-Meier, Nelson-Aalen, Log-rank Test, ...
- Semi-parametric: Cox, Aalen, ...
- Parametric: General Hazard (sucessor of [`HazReg.jl`](https://github.com/FJRubio67/HazReg.jl)), General Odds, ...
- Frailties ? Dont know yet. 

A few notes on details : 

1) For all the semi-paramtric and parametric models, we need to provide an interface that clearly gives the coefficients, eventually statistics about them (significances ? others ?). We might leverage the standard modeling interface from `StatsBase.jl`/`StatsAPI.jl`/`StatsModels.jl`. 
2) The solver used to fit the model, when needed, could be left to the user by leveraging the interface from `Optimization.jl`
3) Through a junction with [`NetSurvival.jl`](https://github.com/JuliaSurv/NetSurvival.jl), the same kind of models would be fittable on net survival data, i.e. with a population mortality offset.

This is still work to be done of course. 