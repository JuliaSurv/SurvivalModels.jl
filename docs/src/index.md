```@meta
CurrentModule = SurvivalModels
```

## Introduction

The [`SurvivalModels.jl`](https://github.com/JuliaSurv/SurvivalModels.jl) package, part of the JuliaSurv ecosystem, provides the necessary tools to perform estimation and analysis of survival data. In contains three categories of models: Non-parametric ones, semi-parametric ones, and fully parametric models. It also contains tests of hypothesis and other features. 

The [`SurvivalModels.jl`](https://github.com/JuliaSurv/SurvivalModels.jl) package is part of the `JuliaSurv` ecosystem around survival analysis. 


Documentation for [SurvivalModels].

In this documentation, we can cite stuff from the `references.bib` file like that : [cox1972regression](@cite). 

This documentation shall cover all the content of the package, which is not the case yet. 


## Features

* Non-parametric modelling: 
    * Kaplan-Meier
    * Log-rank tests, including stratified versions 
* Semi-Parametric modelling: 
    * Cox
* Parametric modelling: 
    * General Hazard models

In particular, our Cox implementation is *fast*, compared to off-the-shelf Julia and R equivalents. 

## Installation

The package is not yet available on Julia's general registry, and thus can be installed through the following command:

```julia
using Pkg
Pkg.add("https://github.com/JuliaSurv/SurvivalModels.jl.git")
```



# Index

```@index
```

# References

```@bibliography
Pages = ["index.md"]
Canonical = false
```