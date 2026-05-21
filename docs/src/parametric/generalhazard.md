# General Hazard Models

## Hazard and cumulative hazard functions

The hazard and the cumulative hazard functions play a crucial role in survival analysis. These functions define the likelihood function in the presence of censored observations. Thus, they are important in many context. For more information about these functions, see [Short course on Parametric Survival Analysis
](https://github.com/FJRubio67/ShortCourseParamSurvival).

In Julia, hazard and cumulative hazard functions can be fetched through the `hazard(dist, t)` and `cumhaz(dist, t)` functions from `SurvivalDistributions.jl`, and can be aplied to any distributions complient with `Distributions.jl`'s API. Note that `SurvivalDistributions.jl` also contains a few more distributions relevant to survival analysis. See also the (deprecated) [HazReg.jl Julia Package](https://github.com/FJRubio67/HazReg.jl). 

Here are a few plots of hazard curves for some known distributions: 

```@example 1
using Distributions, Plots, StatsBase, SurvivalDistributions
function hazard_cumhazard_plot(dist, distname; tlims=(0,10))
      plt1 = plot(t -> hazard(dist, t),
            xlabel = "x", ylabel = "Hazard", title = "$distname distribution",
            xlims = tlims, xticks = tlims[1]:1:tlims[2], label = "",
            xtickfont = font(16, "Courier"), ytickfont = font(16, "Courier"),
            xguidefontsize=18, yguidefontsize=18, linewidth=3,
            linecolor = "blue")
      plt2 = plot(t -> cumhazard(dist, t),
            xlabel = "x", ylabel = "Cumulative Hazard", title = "$distname distribution",
            xlims = tlims, xticks = tlims[1]:1:tlims[2], label = "",
            xtickfont = font(16, "Courier"), ytickfont = font(16, "Courier"),
            xguidefontsize=18, yguidefontsize=18, linewidth=3,
            linecolor = "blue")
      return plot(plt1, plt2)
end
```

### LogNormal

```@example 1
hazard_cumhazard_plot(LogNormal(0.5, 1), "LogNormal")
```

### LogLogistic

```@example 1
hazard_cumhazard_plot(Distributions.LogLogistic(1, 0.5), "LogLogistic")
```

### Weibull

```@example 1
hazard_cumhazard_plot(Weibull(3, 0.5), "Weibull")
```

### Gamma

```@example 1
hazard_cumhazard_plot(Gamma(3, 0.5), "Gamma")
```

## General Hazard Models
The GH model is formulated in terms of the hazard structure

```math
h(t; \alpha, \beta, \theta, {\bf x}) = h_0\left(t  \exp\{\tilde{\bf x}^{\top}\alpha\}; \theta\right) \exp\{{\bf x}^{\top}\beta\}.
```

where ``{\bf x}\in{\mathbb R}^p`` are the covariates that affect the hazard level; ``\tilde{\bf x} \in {\mathbb R}^q`` are the covariates the affect the time level (typically ``\tilde{\bf x} \subset {\bf x}``); ``\alpha \in {\mathbb R}^q`` and ``\beta \in {\mathbb R}^p`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

This hazard structure leads to an identifiable model as long as the baseline hazard is not a hazard associated to a member of the Weibull family of distributions [chen:2001](@cite). 


```@docs
SurvivalModels.GeneralHazardModel
GeneralHazard
```

### Accelerated Failure Time (AFT) model
The AFT model is formulated in terms of the hazard structure

```math
h(t; \beta, \theta, {\bf x}) = h_0\left(t  \exp\{{\bf x}^{\top}\beta\}; \theta\right) \exp\{{\bf x}^{\top}\beta\}.
```

where ``{\bf x}\in{\mathbb R}^p`` are the available covariates; ``\beta \in {\mathbb R}^p`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

```@docs
AcceleratedFaillureTime
```

### Proportional Hazards (PH) model
The PH model is formulated in terms of the hazard structure

```math
h(t; \beta, \theta, {\bf x}) = h_0\left(t ; \theta\right) \exp\{{\bf x}^{\top}\beta\}.
```

where ``{\bf x}\in{\mathbb R}^p`` are the available covariates; ``\beta \in {\mathbb R}^p`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

```@docs
ProportionalHazard
```

### Accelerated Hazards (AH) model
The AH model is formulated in terms of the hazard structure

```math
h(t; \alpha, \theta, \tilde{\bf x}) = h_0\left(t \exp\{\tilde{\bf x}^{\top}\alpha\}; \theta\right) .
```

where ``\tilde{\bf x}\in{\mathbb R}^q`` are the available covariates; ``\alpha \in {\mathbb R}^q`` are the regression coefficients; and ``\theta \in \Theta`` is the vector of parameters of the baseline hazard ``h_0(\cdot)``.

```@docs
AcceleratedHazard
```

### Available baseline hazards
The current version of the `simGH` command implements the following parametric baseline hazards for the models discussed in the previous section.

- [Power Generalised Weibull](http://rpubs.com/FJRubio/PGW) (PGW) distribution.

- [Exponentiated Weibull](http://rpubs.com/FJRubio/EWD) (EW) distribution.

- [Generalised Gamma](http://rpubs.com/FJRubio/GG) (GenGamma) distribuiton.

- [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) (Gamma) distribution.

- [Lognormal](https://en.wikipedia.org/wiki/Log-normal_distribution) (LogNormal) distribution.

- [Log-logistic](https://en.wikipedia.org/wiki/Log-logistic_distribution) (LogLogistic) distribution.

- [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) (Weibull) distribution. (only for AFT, PH, and AH models)


### Prediction interface

Once a `GeneralHazardModel` is fitted (or directly constructed), you can evaluate per-subject cumulative hazards and survival probabilities at user-supplied times. The four hazard structures share the same closed-form expression via the unified representation

```math
H(t \,|\, x) = H_0\!\left(t \cdot c_1(x)\right) \cdot c_2(x)
```

where ``H_0`` is the cumulative hazard of the baseline distribution and ``(c_1, c_2)`` are the method-specific time- and hazard-scale multipliers (`c1`/`c2` in the code). The survival is ``S(t \,|\, x) = \exp(-H(t \,|\, x))``.

```julia
predict(model, :survival)              # length-n vector, each subject at own Tᵢ
predict(model, :expected)              # length-n vector of Λᵢ(Tᵢ)
predict(model, :survival, t)           # length-n vector at scalar t
predict(model, :expected, t)
predict(model, :survival, ts)          # n × length(ts) matrix
predict(model, :expected, ts)
```

The default no-arg form (`predict(model)` or `predict(model, :survival)`) evaluates each subject at their own observed time ``T_i``, matching the convention used by the Cox interface.

#### Predict on new data

Each prediction also accepts a `newdata::DataFrame` argument. The fit's stored formula(s) are re-applied to `newdata` to rebuild the design matrices ``X_1``, ``X_2``, so `newdata` must contain every predictor column referenced in the original `@formula(...)`. For models fit with `fit(GHM, formula, df)` (one formula), the same formula is stored twice and used for both ``X_1`` and ``X_2``; for `fit(GeneralHazard, formula1, formula2, df)` the two are stored separately.

```julia
predict(model, :survival, newdata, t)        # length-n_new at scalar t
predict(model, :expected, newdata, t)
predict(model, :survival, newdata, ts)       # n_new × length(ts) matrix
predict(model, :expected, newdata, ts)
```

Newdata predict **requires** an explicit time argument — there is no "own time" default for arbitrary new subjects. Models built directly via the positional constructor (without the `formula1` / `formula2` keyword arguments) do not have stored formulas and will error on newdata predict.

```@docs
SurvivalModels.predict_expected(::SurvivalModels.GeneralHazardModel)
SurvivalModels.predict_survival(::SurvivalModels.GeneralHazardModel)
```

### Simulating times to event from a general hazard structure with `simGH`

The simGH command from the `HazReg.jl` Julia package allows one to simulate times to event from the following models:

- General Hazard (GH) model [chen:2001](@cite) [rubio:2019](@cite).
- Accelerated Failure Time (AFT) model [kalbfleisch:2011](@cite).
- Proportional Hazards (PH) model [cox:1972](@cite).
- Accelerated Hazards (AH) model [chen:2000](@cite).

A description of these hazard models is presented below as well as the available baseline hazards.

```@docs
simGH
```


## Illustrative example
In this example, we simulate ``n=1,000`` times to event from the GH, PH, AFT, and AH models with PGW baseline hazards, using the `simGH()` function. This functionality was ported from [HazReg.jl](https://github.com/FJRubio67/HazReg.jl) 

### PGW-GH model

```@example 1
using SurvivalModels, Distributions, DataFrames, Random, SurvivalDistributions
using SurvivalModels: simGH

# Simulte design matrices
n = 100
Random.seed!(123)
des = randn(n, 2)
des_t = randn(n, 2)

# True parameters
theta0 = [0.1, 2.0, 5.0]
alpha0 = [0.5, 0.8]
beta0 = [-0.5, 0.75]

# Construct the model directly (no optimization)
model = GeneralHazard(zeros(n), trues(n), 
    PowerGeneralizedWeibull(theta0...),
    des, des_t, alpha0, beta0)

# Simulate event times
simdat = simGH(n, model)

# Administrative censoring. 
cens = 10
status = simdat .< cens
simdat = min.(simdat, cens)

# Model fit from dataframe interface. 
df = DataFrame(time=simdat, status=status, x1=des[:,1], x2=des[:,2], z1=des_t[:,1], z2=des_t[:,2])
model = fit(GeneralHazard{PowerGeneralizedWeibull}, 
    @formula(Surv(time, status) ~ x1 + x2), 
    @formula(Surv(time, status) ~ z1 + z2), 
    df)

result = DataFrame(
    Parameter = ["θ₁", "θ₂", "θ₃", "α₁", "α₂","β₁", "β₂"],
    True      = vcat(theta0, alpha0, beta0),
    Fitted    = vcat(params(model.baseline)..., model.α, model.β)
)
```

Of course, increasing hte numebr of observations would increase the quality of the fitted values. You can also use "subset" models (PH, AH, AFT) through the convenient constructors as follows: 

### PGW-PH model

```@example 1
model = ProportionalHazard(zeros(n), trues(n), 
    PowerGeneralizedWeibull(theta0...),
    des, zeros(n,0),  # X2 is empty for PH
    zeros(0), beta0
)

# Simulate event times and censor them
simdat = simGH(n, model)
cens = 10
status = simdat .< cens
simdat = min.(simdat, cens)


# Build the model and fit it: 
df = DataFrame(time=simdat, status=status, x1=des[:,1], x2=des[:,2])
model = fit(ProportionalHazard{PowerGeneralizedWeibull}, 
    @formula(Surv(time, status) ~ x1 + x2), df)

result = DataFrame(
    Parameter = ["θ₁", "θ₂", "θ₃", "β₁", "β₂"],
    True      = vcat(theta0, beta0),
    Fitted    = vcat(params(model.baseline)..., model.β)
)
```


### PGW-AFT model

```@example 1

# Construct the model directly (no optimization)
model = AcceleratedFaillureTime(
    zeros(n), trues(n), PowerGeneralizedWeibull(theta0...),
    des, zeros(n,0),  # X2 is empty for AFT
    zeros(0), beta0
)

# Simulate event times
simdat = simGH(n, model)

# Censoring
cens = 10
status = simdat .< cens
simdat = min.(simdat, cens)

df = DataFrame(time=simdat, status=status, x1=des[:,1], x2=des[:,2])
model = fit(AcceleratedFaillureTime{PowerGeneralizedWeibull}, 
    @formula(Surv(time, status) ~ x1 + x2), df)

result = DataFrame(
    Parameter = ["θ₁", "θ₂", "θ₃", "β₁", "β₂"],
    True      = vcat(theta0, beta0),
    Fitted    = vcat(params(model.baseline)..., model.β)
)
```



### PGW-AH model

```@example 1
# Construct the model directly (no optimization)
model = AcceleratedHazard(zeros(n), trues(n), 
    PowerGeneralizedWeibull(theta0...),
    zeros(n,0), des_t,  # X1 is empty for AH
    alpha0, zeros(0)
)

# Simulate event times
simdat = simGH(n, model)
cens = 10
status = simdat .< cens
simdat = min.(simdat, cens)

df = DataFrame(time=simdat, status=status, z1=des_t[:,1], z2=des_t[:,2])
model = fit(AcceleratedHazard{PowerGeneralizedWeibull}, 
    @formula(Surv(time, status) ~ z1 + z2), df)

result = DataFrame(
    Parameter = ["θ₁", "θ₂", "θ₃", "α₁", "α₂"],
    True      = vcat(theta0, alpha0),
    Fitted    = vcat(params(model.baseline)..., model.α)
)
```

```@bibliography
Pages = ["generalhazard.md"]
Canonical = false
```