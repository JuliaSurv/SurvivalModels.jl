# SurvivalModels.jl

*Survival analysis models and utilities, written in Julia.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/dev/)
[![Build Status](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.html)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`SurvivalModels.jl` is part of the [JuliaSurv](https://github.com/JuliaSurv) ecosystem. It provides a consistent interface for nonparametric estimation and testing, semiparametric Cox regression, and fully parametric survival models.

## Getting started

The package is available from Julia's General registry:

```julia
using Pkg
Pkg.add("SurvivalModels")
```

Then load it with:

```julia
using SurvivalModels
```

## Available features

### Nonparametric methods

- Kaplan–Meier estimation, including Greenwood confidence intervals.
- Survival and cumulative-hazard prediction at observed or user-supplied times.
- Two- and multi-group log-rank tests.
- Stratified log-rank tests with tied-event variance correction.

### Semiparametric models

- Cox proportional hazards regression through the formula interface.
- Breslow baseline cumulative-hazard estimation.
- Coefficients, standard errors, covariance matrices, confidence intervals, coefficient tables, log partial likelihood, AIC, AICc, and BIC.
- Prediction of linear predictors, relative risks, term contributions, cumulative hazards, and survival probabilities.
- Prediction on new data using the schema stored during fitting.

### Parametric models

- General Hazard (GH), Proportional Hazards (PH), Accelerated Failure Time (AFT), and Accelerated Hazards (AH) model structures.
- Flexible continuous baseline distributions through the `Distributions.jl` interface.
- Formula-based fitting, statistical inference, information criteria, and coefficient tables.
- Cumulative-hazard and survival prediction for training or new data.
- Simulation from GH, PH, AFT, and AH models.

### Model evaluation and interfaces

- Harrell's concordance index for Cox models.
- IPCW Brier scores and integrated Brier scores for Cox and parametric models.
- Standard `fit`, `predict`, `coef`, `vcov`, `confint`, and related interfaces from `StatsBase.jl`, `StatsAPI.jl`, and `StatsModels.jl`.
- `DataFrames.jl` and `@formula(Surv(time, status) ~ predictors)` workflows.

See the [stable documentation](https://JuliaSurv.github.io/SurvivalModels.jl/stable/) for examples and the complete API.

## Contributions are welcome

Questions, bug reports, feature requests, and pull requests are welcome through the repository's [issue tracker](https://github.com/JuliaSurv/SurvivalModels.jl/issues). General collaborative guidelines are available from [ColPrac](https://github.com/SciML/ColPrac).
