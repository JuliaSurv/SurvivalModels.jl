# Model classes

This page documents a family of fully parametric survival models built on a shared *general hazard* (GH) structure,

```math
h(t \mid {\bf x}) = h_0\!\left(t \exp\{\tilde{\bf x}^{\top}\alpha\};\, \theta\right) \exp\{{\bf x}^{\top}\beta\},
```

in which a baseline hazard ``h_0(\cdot;\theta)`` is modulated on both the time scale (through ``\tilde{\bf x}^{\top}\alpha``) and the hazard scale (through ``{\bf x}^{\top}\beta``). The four hazard structures below differ only in which linear predictors enter the baseline: `GeneralHazard` is the general form, and `ProportionalHazard`, `AcceleratedFaillureTime`, and `AcceleratedHazard` constrain it. Each can be constructed directly from parameters or fitted from data via the `fit` interface (see [Illustrative example](@ref)). The [baseline hazards](@ref "Baseline hazards") that supply ``h_0`` are listed on the next page; any continuous distribution complying with the `Distributions.jl` API can serve as ``h_0``.

## General Hazard

The GH model is formulated in terms of the hazard structure

```math
h(t; \alpha, \beta, \theta, {\bf x}) = h_0\left(t  \exp\{\tilde{\bf x}^{\top}\alpha\}; \theta\right) \exp\{{\bf x}^{\top}\beta\}.
```

where ``{\bf x}\in{\mathbb R}^p`` are the covariates that affect the hazard level; ``\tilde{\bf x} \in {\mathbb R}^q`` are the covariates that affect the time level (typically ``\tilde{\bf x} \subset {\bf x}``); ``\alpha \in {\mathbb R}^q`` and ``\beta \in {\mathbb R}^p`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

This hazard structure leads to an identifiable model as long as the baseline hazard is not a hazard associated with a member of the Weibull family of distributions [chen:2001](@cite).

```@docs
SurvivalModels.GeneralHazardModel
GeneralHazard
```

## Proportional Hazards

The PH model is formulated in terms of the hazard structure

```math
h(t; \beta, \theta, {\bf x}) = h_0\left(t ; \theta\right) \exp\{{\bf x}^{\top}\beta\}.
```

where ``{\bf x}\in{\mathbb R}^p`` are the available covariates; ``\beta \in {\mathbb R}^p`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

```@docs
ProportionalHazard
```

## Accelerated Failure Time

The AFT model is formulated in terms of the hazard structure

```math
h(t; \beta, \theta, {\bf x}) = h_0\left(t  \exp\{{\bf x}^{\top}\beta\}; \theta\right) \exp\{{\bf x}^{\top}\beta\}.
```

where ``{\bf x}\in{\mathbb R}^p`` are the available covariates; ``\beta \in {\mathbb R}^p`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

```@docs
AcceleratedFaillureTime
```

## Accelerated Hazards

The AH model is formulated in terms of the hazard structure

```math
h(t; \alpha, \theta, \tilde{\bf x}) = h_0\left(t \exp\{\tilde{\bf x}^{\top}\alpha\}; \theta\right) .
```

where ``\tilde{\bf x}\in{\mathbb R}^q`` are the available covariates; ``\alpha \in {\mathbb R}^q`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

```@docs
AcceleratedHazard
```
