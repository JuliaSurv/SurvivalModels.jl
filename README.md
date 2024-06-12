# SurvivalModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSurv.github.io/SurvivalModels.jl/dev/)
[![Build Status](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSurv/SurvivalModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSurv/SurvivalModels.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/S/SurvivalModels.html)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


The goal of this package is to provide an iterface to fit on standard survival problems the following models: 

- Non-parametric: Kaplan-Meier, Nelson-Aalen, ...
- Semi-parametric: Cox, Aalen, ...
- Parametric: General Hazard (sucessor of [`HazReg.jl`](https://github.com/FJRubio67/HazReg.jl)), General Odds, ...

Then, through a junction with [`NetSurvival.jl`](https://github.com/JuliaSurv/NetSurvival.jl), the same kind of models wouldbe fittable on net survival data, that is with the population mortality offset.


This is still work to be done of course. 