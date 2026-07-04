# Illustrative example

In this example, we simulate ``n=100`` times to event from the GH, PH, AFT, and AH models with PGW baseline hazards, using the `simulate()` function. This functionality was ported from [HazReg.jl](https://github.com/FJRubio67/HazReg.jl).

## PGW-GH model

```@example 1
using SurvivalModels, Distributions, DataFrames, Random, SurvivalDistributions
using SurvivalModels: simulate

# Simulate design matrices
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
simdat = simulate(n, model)

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

Of course, increasing the number of observations would increase the quality of the fitted values. You can also use "subset" models (PH, AH, AFT) through the convenient constructors as follows:

## PGW-PH model

```@example 1
model = ProportionalHazard(zeros(n), trues(n), 
    PowerGeneralizedWeibull(theta0...),
    des, zeros(n,0),  # X2 is empty for PH
    zeros(0), beta0
)

# Simulate event times and censor them
simdat = simulate(n, model)
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

## PGW-AFT model

```@example 1

# Construct the model directly (no optimization)
model = AcceleratedFaillureTime(
    zeros(n), trues(n), PowerGeneralizedWeibull(theta0...),
    des, zeros(n,0),  # X2 is empty for AFT
    zeros(0), beta0
)

# Simulate event times
simdat = simulate(n, model)

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

## PGW-AH model

```@example 1
# Construct the model directly (no optimization)
model = AcceleratedHazard(zeros(n), trues(n), 
    PowerGeneralizedWeibull(theta0...),
    zeros(n,0), des_t,  # X1 is empty for AH
    alpha0, zeros(0)
)

# Simulate event times
simdat = simulate(n, model)
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
