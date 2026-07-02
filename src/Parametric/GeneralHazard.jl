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

# Negative log-likelihood, parameterized by the flat vector
# `par = [log-baseline params (1:npd); α (npd .+ 1:q); β (npd+q .+ 1:p)]`. The
# fit wraps this in a small closure to optimize; `vcov` differentiates it again
# at the optimum to form the observed information — one definition, two callers.
function _neg_loglik(par, method::AbstractGHMethod, base_T, npd, T, Δ, X1, X2)
    q, p = size(X2, 2), size(X1, 2)
    d = base_T(exp.(par[1:npd])...)
    α = par[npd .+ (1:q)]
    β = par[npd + q .+ (1:p)]
    B = (method isa AHMethod) ? 0.0 : (X1[Δ, :] * β)
    C = c1(method, X1, X2, β, α)
    D = c2(method, X1, X2, β, α)
    return -sum(loghazard.(d, T[Δ] .* C[Δ]) .+ B) + sum(cumhazard.(d, T .* C) .* D)
end

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
struct GeneralHazardModel{Method, B} <: StatsAPI.StatisticalModel
    T::Vector{Float64}
    Δ::Vector{Bool}
    baseline::B
    X1::Matrix{Float64}
    X2::Matrix{Float64}
    α::Vector{Float64}
    β::Vector{Float64}
    formula1::Union{FormulaTerm, Nothing}
    formula2::Union{FormulaTerm, Nothing}
    # Maximized log-likelihood at the fitted parameters, captured directly from
    # the optimizer (the fit *is* the maximization of this quantity, so there is
    # nothing to recompute). `NaN` for models built by the direct constructor,
    # which performs no optimization. Consumed by `loglikelihood`/`aic`/`bic`.
    loglik::Float64

    # Direct constructor: all parameters provided, no optimization
    function GeneralHazardModel(::Method, T, Δ, baseline::B, X1, X2, α, β;
                                formula1::Union{FormulaTerm, Nothing}=nothing,
                                formula2::Union{FormulaTerm, Nothing}=nothing) where {Method<:AbstractGHMethod, B}
        X1 = length(size(X1)) == 2 ? X1 : reshape(X1, :, 1) # X1 and X2 must be matrices.
        X2 = length(size(X2)) == 2 ? X2 : reshape(X2, :, 1)
        return new{Method, B}(
            collect(T), Bool.(Δ), baseline,
            Matrix{Float64}(X1), Matrix{Float64}(X2),
            collect(α), collect(β),
            formula1, formula2, NaN,
        )
    end

    # Existing constructor with optimizer (kept as is)
    function GeneralHazardModel(m::Method, T, Δ, baseline, X1, X2;
                                formula1::Union{FormulaTerm, Nothing}=nothing,
                                formula2::Union{FormulaTerm, Nothing}=nothing) where {Method<:AbstractGHMethod}
        npd, p, q = length(Distributions.params(baseline)), size(X1,2), size(X2,2)
        init = vcat(_initial_baseline_log_params(baseline, T), zeros(p + q))
        base_T = typeof(baseline).name.wrapper
        Δ = Bool.(Δ)
        mloglik(par) = _neg_loglik(par, m, base_T, npd, T, Δ, X1, X2)
        if isnan(mloglik(init))
            error("Initial parameters lead to NaN in log-likelihood. Check your baseline distribution and initial values.")
        end
        res = optimize(mloglik, init, LBFGS())
        par = res.minimizer
        # `mloglik` is the negative log-likelihood, so the optimizer's minimum is
        # exactly `-loglik` at the optimum — cache it rather than recomputing.
        loglik = -res.minimum
        d, α, β = base_T(exp.(par[1:npd])...), par[npd .+ (1:q)], par[npd + q .+ (1:p)]
        return new{Method, typeof(d)}(T, Δ, d, X1, X2, α, β, formula1, formula2, loglik)
    end
end

# Indices of the identified regression coefficients within the concatenated
# coefficient vector `[α (1:q); β (q .+ 1:p)]`, where `p = length(β)` (cols of
# `X1`) and `q = length(α)` (cols of `X2`). Per the `c1`/`c2` table above, PH/AFT
# identify only `β`, AH only `α`, GH both. In the single-formula fit `X2 == X1`,
# so the inactive block is carried but not identified — it must be excluded from
# `dof` (else AIC/BIC inflate) and from the Hessian (else it is singular).
#
# This is the single source of truth for "which parameters are active": `dof`,
# the `vcov` Hessian slice, and `coef` all derive from it.
_active_coef_idx(::PHMethod,  p, q) = (q + 1):(q + p)
_active_coef_idx(::AFTMethod, p, q) = (q + 1):(q + p)
_active_coef_idx(::AHMethod,  p, q) = 1:q
_active_coef_idx(::GHMethod,  p, q) = 1:(q + p)

_method(::Type{GeneralHazardModel{M,B}}) where {M,B} = M()
_baseline(::Type{GeneralHazardModel{M,B}}) where {M,B} = B()
_method(::Type{GeneralHazardModel{M}}) where {M} = M()

# ─────────────────────────────────────────────────────────────────────────────
# StatsAPI fit statistics
#
# `GeneralHazardModel <: StatisticalModel`, so once `loglikelihood`, `dof`, and
# `nobs` are defined the generic `aic`/`aicc`/`bic` from StatsAPI work directly.
# ─────────────────────────────────────────────────────────────────────────────

"""
    loglikelihood(m::GeneralHazardModel)

Maximized log-likelihood of the fitted model, captured from the optimizer at fit
time (`NaN` for models built by the direct, non-optimizing constructor).
"""
StatsAPI.loglikelihood(m::GeneralHazardModel) = m.loglik

"""
    nobs(m::GeneralHazardModel)

Number of observations (event/censoring times) the model was fit to.
"""
StatsAPI.nobs(m::GeneralHazardModel) = length(m.T)

"""
    dof(m::GeneralHazardModel)

Number of free parameters consumed by the fit: the baseline distribution's
parameters plus the regression coefficients that actually enter the hazard for
this model's structure (`β` for PH/AFT, `α` for AH, both for GH).
"""
StatsAPI.dof(m::GeneralHazardModel{M}) where {M} =
    length(Distributions.params(m.baseline)) +
    length(_active_coef_idx(M(), size(m.X1, 2), size(m.X2, 2)))

"""
    coef(m::GeneralHazardModel)

Identified parameters on the scale used for inference:
`[log.(baseline parameters); active regression coefficients]`, where the active
coefficients are `β` (PH/AFT), `α` (AH), or `[α; β]` (GH). This ordering matches
[`vcov`](@ref), so `MvNormal(coef(m), vcov(m))` is a coherent sampling
distribution for the parameter uncertainty (exponentiate the baseline entries to
recover the natural-scale baseline parameters).
"""
function StatsAPI.coef(m::GeneralHazardModel{M}) where {M}
    logθ = log.(collect(Float64, Distributions.params(m.baseline)))
    coefs = vcat(m.α, m.β)[_active_coef_idx(M(), length(m.β), length(m.α))]
    return vcat(logθ, coefs)
end

"""
    vcov(m::GeneralHazardModel)

Covariance of [`coef`](@ref): the inverse observed information, i.e. the inverse
of the Hessian of the fitted negative log-likelihood at the optimum, restricted
to the identified parameters (the inactive coefficient block makes the full
Hessian singular). Computed on demand — LBFGS never forms the Hessian during the
fit, so there is nothing cached to reuse. `stderror` follows from this via the
generic `StatisticalModel` method.
"""
function StatsAPI.vcov(m::GeneralHazardModel{Method,B}) where {Method,B}
    base_T = B.name.wrapper
    npd = length(Distributions.params(m.baseline))
    p, q = length(m.β), length(m.α)
    par = vcat(log.(collect(Float64, Distributions.params(m.baseline))), m.α, m.β)
    nll(θ) = _neg_loglik(θ, Method(), base_T, npd, m.T, m.Δ, m.X1, m.X2)
    H = ForwardDiff.hessian(nll, par)
    idx = vcat(1:npd, npd .+ _active_coef_idx(Method(), p, q))
    return inv(H[idx, idx])
end

"""
    _initial_baseline_log_params(baseline, T) -> Vector{Float64}

Seed the optimizer's log-parameter vector for the baseline distribution
of [`GeneralHazardModel`](@ref). For baselines where `Distributions.fit_mle`
exists and returns strictly positive parameters on the (marginal) event
times, use those as the seed; for the rest, anchor the last (conventionally
scale-like) parameter to `log(median(T))` and leave the rest at `log(1) = 0`.

The zeros-init this replaces puts e.g. `Weibull(1, 1)` on data with event
times in 10²–10⁴, putting the log-likelihood in a NaN region and erroring
out (or, before the NaN check, silently returning `β = 0`). See issue #60.

Censoring is ignored for the seed — it's a starting point, not the final
estimate. The joint optimizer subsequently refines `α`, `β`, and the
baseline together.
"""
_initial_baseline_log_params(baseline::Distribution, T) = _scale_anchored_init(baseline, T)

# Distributions.jl ships `fit_mle` for these four; on positive event-time
# data the fitted params are all positive, so taking `log` is well-defined.
_initial_baseline_log_params(::Weibull,     T) = _log_fit_params(Weibull,     T)
_initial_baseline_log_params(::LogNormal,   T) = _log_fit_params(LogNormal,   T)
_initial_baseline_log_params(::Normal,      T) = _log_fit_params(Normal,      T)
_initial_baseline_log_params(::Exponential, T) = _log_fit_params(Exponential, T)

_log_fit_params(B::Type{<:Distribution}, T) = log.(collect(Float64, Distributions.params(Distributions.fit_mle(B, T))))

_initial_baseline_log_params(::PowerGeneralizedWeibull, T) = [0.0, 0.0, 0.0]

function _scale_anchored_init(baseline::Distribution, T)
    npd = length(Distributions.params(baseline))
    init = zeros(npd)
    npd > 0 && (init[npd] = log(median(T)))
    return init
end

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
    return GeneralHazardModel(_method(GHM), TΔ[:,1], TΔ[:,2], _baseline(GHM), X1, X2;
                              formula1 = f1_applied, formula2 = f2_applied)
end
function StatsBase.fit(::Type{GHM},
    formula::FormulaTerm,
    df::DataFrame) where {GHM <: GeneralHazardModel}
    sdf = schema(df)
    f_applied = apply_schema(formula, sdf)
    X = modelcols(f_applied.rhs, df)
    TΔ = modelcols(f_applied.lhs, df)
    # Use X for both X1 and X2 (PH, AFT, AH models will ignore one of them).
    # Both formulas are set to the same applied formula so predict-on-newdata
    # works regardless of which method (PH/AFT/AH/GH) the user picked.
    return GeneralHazardModel(_method(GHM), TΔ[:,1], TΔ[:,2], _baseline(GHM), X, X;
                              formula1 = f_applied, formula2 = f_applied)
end

# Internal helper: return the per-subject (c1, c2) multipliers for any method,
# given explicit design matrices. Used for both training (m.X1/m.X2) and newdata.
function _c1c2(m::GeneralHazardModel{M,B}, X1, X2) where {M,B}
    method = M()
    return c1(method, X1, X2, m.β, m.α), c2(method, X1, X2, m.β, m.α)
end

# Convenience overload for training data.
_c1c2(m::GeneralHazardModel) = _c1c2(m, m.X1, m.X2)

# Apply the fit's stored formulas to `newdata` and return `(X1_new, X2_new)`.
# Errors if the model was constructed without stored formulas (e.g. via the
# direct positional constructor without the kw `formula1` / `formula2`).
function _build_X_for_newdata(m::GeneralHazardModel, newdata::DataFrame)
    if isnothing(m.formula1) || isnothing(m.formula2)
        error("This GeneralHazardModel was constructed without stored formulas; predict-on-newdata is not available. Re-fit via `fit(GHM, @formula(...), df)` (or the two-formula variant for `GeneralHazard`) so the formulas are captured.")
    end
    X1_new = modelcols(m.formula1.rhs, newdata)
    X2_new = modelcols(m.formula2.rhs, newdata)
    return X1_new, X2_new
end

"""
    predict_expected(m::GeneralHazardModel)
    predict_expected(m::GeneralHazardModel, t::Real)
    predict_expected(m::GeneralHazardModel, ts::AbstractVector)
    predict_expected(m::GeneralHazardModel, newdata::DataFrame, t::Real)
    predict_expected(m::GeneralHazardModel, newdata::DataFrame, ts::AbstractVector)

Per-subject cumulative hazard ``\\Lambda_i(t) = H_0(t\\, c_{1i})\\, c_{2i}``, where ``H_0`` is the
cumulative hazard of the baseline distribution and ``(c_{1i}, c_{2i})`` are the method-specific
time- and hazard-scale multipliers (PH, AFT, AH, GH share the same closed form via the unified
``H(t \\mid x) = H_0(t\\, c_1)\\, c_2`` representation).

Output shape:
- no time argument → length-`n` vector with each subject evaluated at their own
  observed time ``T_i``;
- `t::Real` → length-`n` vector at the scalar time;
- `ts::AbstractVector` → `n × length(ts)` matrix.

With `newdata::DataFrame` the design matrices are rebuilt by applying the fit's stored
formula(s) — `newdata` must contain every predictor column referenced in the original
`@formula(...)`. Newdata predict requires an explicit time argument (no "own time"
default).
"""
function predict_expected(m::GeneralHazardModel)
    cc1, cc2 = _c1c2(m)
    return [(-logccdf(m.baseline, m.T[i] * cc1[i])) * cc2[i] for i in eachindex(m.T)]
end

function predict_expected(m::GeneralHazardModel, t::Real)
    cc1, cc2 = _c1c2(m)
    return (-logccdf.(m.baseline, t .* cc1)) .* cc2
end

function predict_expected(m::GeneralHazardModel, ts::AbstractVector)
    cc1, cc2 = _c1c2(m)
    n = length(cc1)
    out = Matrix{Float64}(undef, n, length(ts))
    @inbounds for j in eachindex(ts)
        out[:, j] = (-logccdf.(m.baseline, ts[j] .* cc1)) .* cc2
    end
    return out
end

function predict_expected(m::GeneralHazardModel, newdata::DataFrame, t::Real)
    X1_new, X2_new = _build_X_for_newdata(m, newdata)
    cc1, cc2 = _c1c2(m, X1_new, X2_new)
    return (-logccdf.(m.baseline, t .* cc1)) .* cc2
end

function predict_expected(m::GeneralHazardModel, newdata::DataFrame, ts::AbstractVector)
    X1_new, X2_new = _build_X_for_newdata(m, newdata)
    cc1, cc2 = _c1c2(m, X1_new, X2_new)
    n = length(cc1)
    out = Matrix{Float64}(undef, n, length(ts))
    @inbounds for j in eachindex(ts)
        out[:, j] = (-logccdf.(m.baseline, ts[j] .* cc1)) .* cc2
    end
    return out
end

"""
    predict_survival(m::GeneralHazardModel)
    predict_survival(m::GeneralHazardModel, t::Real)
    predict_survival(m::GeneralHazardModel, ts::AbstractVector)
    predict_survival(m::GeneralHazardModel, newdata::DataFrame, t::Real)
    predict_survival(m::GeneralHazardModel, newdata::DataFrame, ts::AbstractVector)

Per-subject survival probability ``S_i(t) = \\exp(-\\Lambda_i(t))`` derived from
[`predict_expected`](@ref). Shapes match `predict_expected`; newdata variants
require an explicit time argument.
"""
predict_survival(m::GeneralHazardModel)                     = exp.(-predict_expected(m))
predict_survival(m::GeneralHazardModel, t::Real)            = exp.(-predict_expected(m, t))
predict_survival(m::GeneralHazardModel, ts::AbstractVector) = exp.(-predict_expected(m, ts))
predict_survival(m::GeneralHazardModel, newdata::DataFrame, t::Real)            = exp.(-predict_expected(m, newdata, t))
predict_survival(m::GeneralHazardModel, newdata::DataFrame, ts::AbstractVector) = exp.(-predict_expected(m, newdata, ts))

function StatsBase.predict(m::GeneralHazardModel, type::Symbol=:survival)
    type == :survival && return predict_survival(m)
    type == :expected && return predict_expected(m)
    error("Unsupported predict type `:$type` for GeneralHazardModel. Supported: `:survival`, `:expected`.")
end

function StatsBase.predict(m::GeneralHazardModel, type::Symbol, t::Real)
    type == :survival && return predict_survival(m, t)
    type == :expected && return predict_expected(m, t)
    error("Time-indexed `predict` on GeneralHazardModel supports `:survival` and `:expected`, got `:$type`.")
end

function StatsBase.predict(m::GeneralHazardModel, type::Symbol, ts::AbstractVector)
    type == :survival && return predict_survival(m, ts)
    type == :expected && return predict_expected(m, ts)
    error("Time-indexed `predict` on GeneralHazardModel supports `:survival` and `:expected`, got `:$type`.")
end

function StatsBase.predict(m::GeneralHazardModel, type::Symbol, newdata::DataFrame)
    type in (:survival, :expected) && error("`:$type` on newdata requires a time argument: predict(m, :$type, newdata, t).")
    error("Unsupported predict type `:$type` for GeneralHazardModel on newdata. Supported: `:survival`, `:expected` (with a time argument).")
end

function StatsBase.predict(m::GeneralHazardModel, type::Symbol, newdata::DataFrame, t::Real)
    type == :survival && return predict_survival(m, newdata, t)
    type == :expected && return predict_expected(m, newdata, t)
    error("Time-indexed `predict` on GeneralHazardModel newdata supports `:survival` and `:expected`, got `:$type`.")
end

function StatsBase.predict(m::GeneralHazardModel, type::Symbol, newdata::DataFrame, ts::AbstractVector)
    type == :survival && return predict_survival(m, newdata, ts)
    type == :expected && return predict_expected(m, newdata, ts)
    error("Time-indexed `predict` on GeneralHazardModel newdata supports `:survival` and `:expected`, got `:$type`.")
end

# ─────────────────────────────────────────────────────────────────────────────
# Simulate / evaluate from an explicit baseline + design + coefficients
#
# These methods need no fitted model: pass the hazard structure (`method`), the
# baseline distribution, the design matrices `X1`/`X2` (or a formula + DataFrame),
# and the coefficients `α` (for `X2`) and `β` (for `X1`). They reuse the same
# `c1`/`c2` multipliers as the fitted-model methods, so simulation, survival, and
# cumulative hazard stay consistent with `fit`/`predict`.
# ─────────────────────────────────────────────────────────────────────────────

_as_matrix(X) = X isa AbstractMatrix ? X : reshape(X, :, 1)
_gh_design(formula::FormulaTerm, df) = modelcols(apply_schema(formula, schema(df)).rhs, df)

"""
    simulate(n, method, baseline, X1, X2, α, β; rng=Random.default_rng())
    simulate(n, method, baseline, formula[, formula2], df, α, β; rng=Random.default_rng())
    simulate(n, model::GeneralHazardModel; rng=Random.default_rng())

Simulate `n` times to event from a general-hazard model with hazard structure
`method` (`PHMethod()`/`AFTMethod()`/`AHMethod()`/`GHMethod()`) and `baseline`
distribution. Each of the `n` rows of the design (`X1`/`X2`, or built from
`formula`/`df` via `modelcols` exactly as in `fit`) yields one event time by
inverting that subject's survival `Sᵢ(t)=U`, i.e.
`Tᵢ = quantile(baseline, 1 − (1−Uᵢ)^(1/c2ᵢ)) / c1ᵢ`. `α` are the `X2`
coefficients and `β` the `X1` coefficients (only the ones the structure uses).

The fitted-model form delegates here using the model's own design/coefficients.

References:
* [HazReg original code](https://github.com/FJRubio67/HazReg)
"""
function simulate(n::Integer, method::AbstractGHMethod, baseline, X1::AbstractVecOrMat, X2::AbstractVecOrMat, α, β; rng = Random.default_rng())
    X1 = _as_matrix(X1)
    X2 = _as_matrix(X2)
    cc1 = c1(method, X1, X2, β, α)
    cc2 = c2(method, X1, X2, β, α)
    U = rand(rng, n)
    return quantile.(baseline, 1 .- (1 .- U) .^ (1 ./ cc2)) ./ cc1
end
function simulate(n::Integer, method::AbstractGHMethod, baseline, formula1::FormulaTerm, formula2::FormulaTerm, df, α, β; rng = Random.default_rng())
    simulate(n, method, baseline, _gh_design(formula1, df), _gh_design(formula2, df), α, β; rng)
end
simulate(n::Integer, method::AbstractGHMethod, baseline, formula::FormulaTerm, df, α, β; rng = Random.default_rng()) =
    simulate(n, method, baseline, formula, formula, df, α, β; rng)
simulate(n::Integer, m::GeneralHazardModel{M,B}; rng = Random.default_rng()) where {M,B} =
    simulate(n, M(), m.baseline, m.X1, m.X2, m.α, m.β; rng)

# Deprecated: `simGH` was the original (placeholder) simulation entry point; it is
# now `simulate`. Kept (and still exported) as a thin, warning-emitting alias.
Base.@deprecate simGH(n, m::GeneralHazardModel; rng = Random.default_rng()) simulate(n, m; rng)

# Per-subject cumulative hazard `Λᵢ(t) = H₀(t·c1ᵢ)·c2ᵢ` from an explicit baseline +
# design + coefficients (no fitted model). `t::Real` → length-`n` vector (one per
# design row); `t::AbstractVector` → `n × length(t)` matrix. These mirror the
# fitted-model `predict_expected`/`predict_survival` methods; `predict_survival`
# is `exp(-Λ)`. (Comment-documented to keep the strict docs build's docstring set
# anchored on `simulate`.)
function predict_expected(method::AbstractGHMethod, baseline, X1::AbstractVecOrMat, X2::AbstractVecOrMat, α, β, t::Real)
    X1 = _as_matrix(X1)
    X2 = _as_matrix(X2)
    cc1 = c1(method, X1, X2, β, α)
    cc2 = c2(method, X1, X2, β, α)
    return (-logccdf.(baseline, t .* cc1)) .* cc2
end
function predict_expected(method::AbstractGHMethod, baseline, X1::AbstractVecOrMat, X2::AbstractVecOrMat, α, β, ts::AbstractVector)
    X1 = _as_matrix(X1)
    X2 = _as_matrix(X2)
    cc1 = c1(method, X1, X2, β, α)
    cc2 = c2(method, X1, X2, β, α)
    out = Matrix{Float64}(undef, length(cc1), length(ts))
    @inbounds for j in eachindex(ts)
        out[:, j] = (-logccdf.(baseline, ts[j] .* cc1)) .* cc2
    end
    return out
end
predict_expected(method::AbstractGHMethod, baseline, f1::FormulaTerm, f2::FormulaTerm, df, α, β, t) =
    predict_expected(method, baseline, _gh_design(f1, df), _gh_design(f2, df), α, β, t)
predict_expected(method::AbstractGHMethod, baseline, f::FormulaTerm, df, α, β, t) =
    predict_expected(method, baseline, f, f, df, α, β, t)

# Per-subject survival `Sᵢ(t) = exp(-Λᵢ(t))` from an explicit baseline + design +
# coefficients (no fitted model); shapes match the `predict_expected` pieces forms.
predict_survival(method::AbstractGHMethod, baseline, X1::AbstractVecOrMat, X2::AbstractVecOrMat, α, β, t) =
    exp.(-predict_expected(method, baseline, X1, X2, α, β, t))
predict_survival(method::AbstractGHMethod, baseline, f1::FormulaTerm, f2::FormulaTerm, df, α, β, t) =
    exp.(-predict_expected(method, baseline, f1, f2, df, α, β, t))
predict_survival(method::AbstractGHMethod, baseline, f::FormulaTerm, df, α, β, t) =
    exp.(-predict_expected(method, baseline, f, df, α, β, t))
