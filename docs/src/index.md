```@meta
CurrentModule = SurvivalModels
```

## Introduction

The [`SurvivalModels.jl`](https://github.com/JuliaSurv/SurvivalModels.jl) package, part of the JuliaSurv ecosystem, provides the necessary tools to perform estimation and analysis of survival data. In contains three categories of models: Non-parametric ones, semi-parametric ones, and fully parametric models. It also contains tests of hypothesis and other features:

## Features

* Non-parametric modelling: 
    * Kaplan-Meier
    * Log-rank tests, including stratified versions 
* Semi-Parametric modelling: 
    * Cox
* Parametric modelling: 
    * General Hazard models

In particular, our Cox implementation is *fast*, compared to off-the-shelf Julia and R equivalents. Check it out ! 

If you find the slightest bug or want to discuss addition of other methods, or simply chat, do not hesitate to open an issue on the repository. 

## Installation

The package is available in Julia's `General` registry, and thus can be installed through the following command:

```julia
] add SurvivalModels
```

# Index

```@index
```
