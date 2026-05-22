# SurvivalModels.jl

*A pure-julia take on standard survival analysis modeling.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/dev/)
[![Build Status](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml?query=branch:main)
[![Coverage](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.html)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

The `SurvivalModels.jl` package is part of the `JuliaSurv` survival analysis suite. It provides the necessary tools to perform modeling of survival data, from non-parametric estimators and tests to parametric and semi-parametric models with a unified interface.

# Getting started

The package is available on Julia's General registry, therefore you can use the following to install it:

```julia
] add SurvivalModels
```

# Features / Roadmap

The package targets the following features:

- **Nonparametric**
    - [x] Kaplan-Meier
    - [x] Log-rank test (including stratification)
- **Semi-parametric**
    - [x] Cox (See PR #15)
        - [x] Prediction on new data
    - [ ] Aalen
- **Parametric**
    - [x] General Hazard Models
        - [x] Prediction on new data
    - [ ] Frailties
    - [ ] Mixed models
    - [ ] More generic predictors such as splines
- **Metrics**
    - [x] Harrell's C-index
    - [x] Brier score
    - [x] Integrated Brier score
- [ ] Junction with [`NetSurvival.jl`](https://github.com/JuliaSurv/NetSurvival.jl) to provide the same models on net survival instead of survival (i.e. with a population mortality offset)
- [ ] Something else on this list? Open a PR :)

In terms of interface, we leverage the standard modeling interface from `StatsBase.jl`/`StatsAPI.jl`/`StatsModels.jl`.

Some of the models might not provide all the outputs you need at the moment. Feel free to open an issue to tell us; we'll look at it and add the features if we can :)

# Contributions are welcome

If you want to contribute to the package, ask a question, found a bug, or simply want to chat, do not hesitate to open an issue on this repo. General guidelines on collaborative practices (ColPrac) are available [here](https://github.com/SciML/ColPrac).
