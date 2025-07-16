abstract type AbstractGHMethod end
struct PHMethod  <: AbstractGHMethod end
struct AFTMethod <: AbstractGHMethod end
struct AHMethod  <: AbstractGHMethod end
struct GHMethod  <: AbstractGHMethod end

c1(::GHMethod,  X1, X2, β, α) = exp.(X2 * α)
c1(::PHMethod,  X1, X2, β, α) = ones(size(X1, 1))
c1(::AFTMethod, X1, X2, β, α) = exp.(X1 * β)
c1(::AHMethod,  X1, X2, β, α) = exp.(X2 * α)

c2(::GHMethod,  X1, X2, β, α) = exp.(X1 * β - X2 * α)
c2(::PHMethod,  X1, X2, β, α) = exp.(X1 * β)
c2(::AFTMethod, X1, X2, β, α) = ones(size(X1, 1))
c2(::AHMethod,  X1, X2, β, α) = exp.(-X2 * α)

"""
    GeneralHazardModel{Method, B}

A flexible parametric survival model supporting Proportional Hazards (PH), Accelerated Failure Time (AFT), Accelerated Hazards (AH), and General Hazards (GH) structures.

# Fields
- `T`: Vector of observed times.
- `Δ`: Vector of event indicators (true if event, false if censored).
- `baseline`: Baseline distribution (e.g., `Weibull()`).
- `X1`: Covariate matrix for the first linear predictor (e.g., PH/AFT).
- `X2`: Covariate matrix for the second linear predictor (e.g., AH/GH).
- `α`: Coefficient vector for `X2`.
- `β`: Coefficient vector for `X1`.

# Construction

You can construct a model directly by providing all parameters:
```julia
model = GeneralHazardModel(
    GHMethod(),
    T, Δ, Weibull(1.0, 2.0),
    X1, X2,
    α, β
)
```
or fit it from data using the `fit` interface.

Supported methods: 
- `ProportionalHazard`: For PH models.
- `AcceleratedFaillureTime`: For AFT models.
- `AcceleratedHazard`: For AH models.
- `GeneralHazard`: For full GH models.
"""
struct GeneralHazardModel{Method, B}
    T::Vector{Float64}
    Δ::Vector{Bool}
    baseline::B
    X1::Matrix{Float64}
    X2::Matrix{Float64}
    α::Vector{Float64}
    β::Vector{Float64}

    # Direct constructor: all parameters provided, no optimization
    function GeneralHazardModel(::Method, T, Δ, baseline::B, X1, X2, α, β) where {Method<:AbstractGHMethod, B}
        X1 = length(size(X1)) == 2 ? X1 : reshape(X1, :, 1) # X1 and X2 must be matrices. 
        X2 = length(size(X2)) == 2 ? X2 : reshape(X2, :, 1)
        return new{Method, B}(
            collect(T), Bool.(Δ), baseline,
            Matrix{Float64}(X1), Matrix{Float64}(X2),
            collect(α), collect(β)
        )
    end

    # Existing constructor with optimizer (kept as is)
    function GeneralHazardModel(m::Method, T, Δ, baseline, X1, X2) where {Method<:AbstractGHMethod}
        npd, p, q = length(Distributions.params(baseline)), size(X1,2), size(X2,2)
        init = zeros(npd+p+q)
        base_T = typeof(baseline).name.wrapper
        Δ = Bool.(Δ)
        function mloglik(par::Vector)
            d, α, β = base_T(exp.(par[1:npd])...), par[npd .+ (1:q)], par[npd + q .+ (1:p)]
            B = (Method == AHMethod) ? 0.0 : (X1[Δ,:] * β)
            C = c1(m, X1, X2, β, α)
            D = c2(m, X1, X2, β, α)
            return  -sum(loghazard.(d, T[Δ] .* C[Δ]) .+ B) + sum(cumhazard.(d, T .* C) .* D)
        end
        par = optimize(mloglik, init, method=LBFGS()).minimizer
        d, α, β = base_T(exp.(par[1:npd])...), par[npd .+ (1:q)], par[npd + q .+ (1:p)]
        return new{Method, typeof(d)}(T, Δ, d, X1, X2, α, β)
    end
end

_method(::Type{GeneralHazardModel{M,B}}) where {M,B} = M()
_baseline(::Type{GeneralHazardModel{M,B}}) where {M,B} = B()
_method(::Type{GeneralHazardModel{M}}) where {M} = M()

"""
    ProportionalHazard(T, Δ, baseline, X1, X2)
    fit(ProportionalHazard, @formula(Surv(T, Δ) ~ x1 + x2), df)

Fit a Proportional Hazards (PH) model with a specified baseline distribution and covariates.

# Hazard function

```math
h(t \\,|\\, x) = h_0(t) \\exp(x^\\top \\beta)
```

- `T`: Vector of observed times.
- `Δ`: Vector of event indicators (1=event, 0=censored).
- `baseline`: Baseline distribution (e.g., Weibull()).
- `X1`, `X2`: Covariate matrices (only `X1` is used in PH).

You can also use the `fit()` interface with a formula and DataFrame.
"""
const ProportionalHazard{B} = GeneralHazardModel{PHMethod, B}
ProportionalHazard(args...) = GeneralHazardModel(PHMethod(),  args...)

"""
    AcceleratedFaillureTime(T, Δ, baseline, X1, X2)
    fit(AcceleratedFaillureTime, @formula(Surv(T, Δ) ~ x1 + x2), df)

Fit an Accelerated Failure Time (AFT) model with a specified baseline distribution and covariates.

# Hazard function

```math
h(t \\,|\\, x) = h_0\\left(t \\exp(x^\\top \\beta)\\right) \\exp(x^\\top \\beta)
```

- `T`: Vector of observed times.
- `Δ`: Vector of event indicators (1=event, 0=censored).
- `baseline`: Baseline distribution (e.g., Weibull()).
- `X1`, `X2`: Covariate matrices (only `X1` is used in AFT).

You can also use the `fit()` interface with a formula and DataFrame.
"""
const AcceleratedFaillureTime{B} = GeneralHazardModel{AFTMethod, B}
AcceleratedFaillureTime(args...) = GeneralHazardModel(AFTMethod(), args...)

"""
    AcceleratedHazard(T, Δ, baseline, X1, X2)
    fit(AcceleratedHazard, @formula(Surv(T, Δ) ~ x1 + x2), df)

Fit an Accelerated Hazard (AH) model with a specified baseline distribution and covariates.

# Hazard function

```math
h(t \\,|\\, z) = h_0\\left(t \\exp(z^\\top \\alpha)\\right)
```

- `T`: Vector of observed times.
- `Δ`: Vector of event indicators (1=event, 0=censored).
- `baseline`: Baseline distribution (e.g., Weibull()).
- `X1`, `X2`: Covariate matrices (only `X2` is used in AH).

You can also use the `fit()` interface with a formula and DataFrame.
"""
const AcceleratedHazard{B} = GeneralHazardModel{AHMethod, B}
AcceleratedHazard(args...) = GeneralHazardModel(AHMethod(),  args...)

"""
    GeneralHazard(T, Δ, baseline, X1, X2)
    fit(GeneralHazard, @formula(Surv(T, Δ) ~ x1 + x2), @formula(Surv(T, Δ) ~ z1 + z2), df)
    fit(GeneralHazard, @formula(Surv(T, Δ) ~ x1 + x2), df)

Fit a General Hazard (GH) model with a specified baseline distribution and covariates.

# Hazard function

```math
h(t \\,|\\, x, z) = h_0\\left(t \\exp(z^\\top \\alpha)\\right) \\exp(x^\\top \\beta)
```

Maximum likelihood estimation in General Hazards models using provided `baseline` distribution, provided hazard structure (through the `method` argument), provided design matrices.. 

Parameters `T,Δ` represent observed times and statuses, while `X1, X2` should contain covariates. The number of columns in design matrices can be zero. 

Hazard structures are defined by the method, which should be `<:AbstractGHMethod`, available possibilities are `PHMethod()`, `AFTMethod()`, `AHMethod()` and `GHMethod()`.

The baseline distribution should be provided as a `<:Distributions.ContinuousUnivariateDistribution` object from `Distributions.jl` or compliant, e.g. from `SurvivalDistributions.jl`.

- `T`: Vector of observed times.
- `Δ`: Vector of event indicators (1=event, 0=censored).
- `baseline`: Baseline distribution (e.g., Weibull()).
- `X1`, `X2`: Covariate matrices.

You can also use the `fit()` interface with:
- Two formulas (for `X1` and `X2`): for full GH models.
- One formula: for PH, AFT, or AH models (the unused matrix will be ignored).

# Example: Direct usage

```julia
using SurvivalModels, Distributions, Optim
T = [2.0, 3.0, 4.0, 5.0, 8.0]
Δ = [1, 1, 0, 1, 0]
X1 = [1.0 2.0; 2.0 1.0; 3.0 1.0; 4.0 2.0; 5.0 1.0]
X2 = [1.0 0.0; 0.0 1.0; 1.0 1.0; 0.0 0.0; 1.0 1.0]
model = GeneralHazard(T, Δ, Weibull, X1, X2)
```

# Example: Using the fit() interface

```julia
using SurvivalModels, DataFrames, Distributions, Optim, StatsModels
df = DataFrame(time=T, status=Δ, x1=X1[:,1], x2=X1[:,2], z1=X2[:,1], z2=X2[:,2])
model = fit(GeneralHazard, @formula(Surv(time, status) ~ x1 + x2), @formula(Surv(time, status) ~ z1 + z2), df)
# Or for PH/AFT/AH models:
model_ph = fit(ProportionalHazard, @formula(Surv(time, status) ~ x1 + x2), df)
```

References: 
* [Link to my reference so that people understand what it is](https://myref.com)
"""
const GeneralHazard{B} = GeneralHazardModel{GHMethod,  B}
GeneralHazard(args...) = GeneralHazardModel(GHMethod(),  args...)


function StatsBase.fit(::Type{GHM},
    formula1::FormulaTerm,
    formula2::FormulaTerm,
    df::DataFrame) where {GHM <: GeneralHazardModel}
    sdf = schema(df)
    f1_applied = apply_schema(formula1, sdf)
    f2_applied = apply_schema(formula2, sdf)
    X1 = modelcols(f1_applied.rhs, df)
    X2 = modelcols(f2_applied.rhs, df)
    TΔ = modelcols(f1_applied.lhs, df)
    return GeneralHazardModel(_method(GHM), TΔ[:,1], TΔ[:,2], _baseline(GHM), X1, X2)
end
function StatsBase.fit(::Type{GHM},
    formula::FormulaTerm,
    df::DataFrame) where {GHM <: GeneralHazardModel}
    sdf = schema(df)
    f_applied = apply_schema(formula, sdf)
    X = modelcols(f_applied.rhs, df)
    TΔ = modelcols(f_applied.lhs, df)
    # Use X for both X1 and X2 (PH, AFT, AH models will ignore one of them)
    return GeneralHazardModel(_method(GHM), TΔ[:,1], TΔ[:,2], _baseline(GHM), X, X)
end

"""
    simGH(n, model::GeneralHazardModel)

This function simulate times to event from a general hazard model, whatever the structure it has (AH, AFT, PH, GH), and whatever its baseline distribution. 

Returns a vector containing the simulated times to event

References: 
* [HazReg original code](https://github.com/FJRubio67/HazReg) 
"""
function simGH(n, m::GeneralHazardModel{M,B}) where {M,B}
    args = (M(), m.X1, m.X2, m.β, m.α)
    p0 = 1 .- exp.(log.(1 .- rand(n)) ./ c2(args...))
    return quantile.(m.baseline,p0) ./ c1(args...)
end