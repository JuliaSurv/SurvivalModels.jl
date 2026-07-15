"""
    KaplanMeier(T, Δ)
    fit(KaplanMeier, @formula(Surv(T, Δ) ~ 1), df)

Efficient Kaplan-Meier estimator.

# Mathematical Description

Suppose we observe ``n`` individuals, with observed times ``T_1, T_2, \\ldots, T_n`` and event indicators ``\\Delta_1, \\Delta_2, \\ldots, \\Delta_n`` (``\\Delta_i = 1`` if the event occurred, ``0`` if censored).

Let ``t_1 < t_2 < \\cdots < t_k`` be the ordered unique event times.

- ``d_j``: number of events at time ``t_j``
- ``Y_j``: number of individuals at risk just before ``t_j``

The **Kaplan-Meier estimator** of the survival function ``S(t)`` is:

```math
\\hat{S}(t) = \\prod_{t_j \\leq t} \\left(1 - \\frac{d_j}{Y_j}\\right)
```

This product runs over all event times ``t_j`` less than or equal to ``t``.

The Greenwood estimator for the variance of ``\\hat{S}(t)`` is:

```math
\\widehat{\\mathrm{Var}}[\\hat{S}(t)] = \\hat{S}(t)^2 \\sum_{t_j \\leq t} \\frac{d_j}{Y_j (Y_j - d_j)}
```

# Arguments
- `T`: Vector of event or censoring times.
- `Δ`: Event indicator vector (`1` if event, `0` if censored).

# Stores
- `t`: Sorted unique event times.
- `∂N`: Number of uncensored deaths at each time point.
- `Y`: Number of at risk individuals at each time point.
- `∂Λ`: Increments of cumulative hazard.
- `∂σ`: Greenwood variance increments.

# Example: Direct usage

```julia
using SurvivalModels
T = [2, 3, 4, 5, 8]
Δ = [1, 1, 0, 1, 0]
km = KaplanMeier(T, Δ)
```

# Example: Using the fit() interface

```julia
using SurvivalModels, DataFrames, StatsModels
df = DataFrame(time=T, status=Δ)
km2 = fit(KaplanMeier, @formula(Surv(time, status) ~ 1), df)
```
"""
struct KaplanMeier{T}
    t::Vector{T}
    ∂N::Vector{Int64}
    Y::Vector{Int64}
    ∂Λ::Vector{T}
    ∂σ::Vector{T}
    function KaplanMeier(T::AbstractVector, Δ::AbstractVector)
        @assert length(T) == length(Δ)
        o = sortperm(T)
        To = T[o].*1.0 # Convert to Float64 for consistency
        Δo = Bool.(Δ[o])
        t = unique(To)
        N, n = length(To), length(t)
        ∂N, Y, ∂Λ, ∂σ = zeros(Int64, n), zeros(Int64, n), zero(t), zero(t)
        j = 1
        at_risk = N
        for i in 1:n
            ti = t[i]
            # Compute size and n_events of the risk set:
            rs_size, rs_events = 0, 0
            while j <= N && To[j] == ti
                rs_size += 1
                rs_events += Δo[j]
                j += 1
            end
            # Imput this data in the processes: 
            ∂N[i] = rs_events
            Y[i] = at_risk
            ∂Λ[i] = at_risk == 0 ? 0 : rs_events / at_risk
            ∂σ[i] = (at_risk == 0 || at_risk == rs_events) ? 0 : rs_events / (at_risk * (at_risk - rs_events))
            # Decrease the number of people at risk by the risk set size: 
            at_risk -= rs_size
        end
        new{eltype(To)}(t, ∂N, Y, ∂Λ, ∂σ)
    end
end

function StatsBase.fit(::Type{T}, formula::FormulaTerm, df::DataFrame) where {T<:KaplanMeier}
    # `KaplanMeier` estimates a single survival curve; right-hand-side variables
    # would silently be ignored (it is not a stratified estimator), so reject them
    # rather than return a misleading pooled curve. See issue on stratification.
    isempty(StatsModels.termvars(formula.rhs)) || throw(ArgumentError(
        "KaplanMeier estimates a single survival curve and does not support " *
        "right-hand-side variables; use `Surv(t, s) ~ 1`. Fit separate curves per " *
        "group, or use `LogRankTest` to compare groups."))
    # `schema(formula, df)` reads only the columns the formula references, then
    # `apply_schema` turns the `Surv(t, s)` LHS into a response term whose
    # `modelcols` returns an `n×2` matrix: column 1 the times, column 2 the event
    # indicator. Coerce the indicator to `Bool` for the constructor.
    formula_applied = apply_schema(formula, schema(formula, df))
    resp = modelcols(formula_applied.lhs, df)
    return KaplanMeier(resp[:, 1], Bool.(resp[:, 2]))
end

# Survival estimate Ŝ(t)
(S::KaplanMeier)(t) = prod((1 - S.∂Λ[i] for i in eachindex(S.t) if S.t[i] < t); init=1.0)

"""
    predict(km::KaplanMeier)                       # = predict(km, :survival)
    predict(km::KaplanMeier, type::Symbol)         # vector at km.t
    predict(km::KaplanMeier, type::Symbol, t)      # scalar (step-function eval)
    predict(km::KaplanMeier, type::Symbol, ts::AbstractVector)  # vector at ts

Survival-function (`type = :survival`) or cumulative-hazard (`type =
:cumhazard`) predictions from a fitted Kaplan-Meier estimator. With no
time argument the result has one entry per stored event/censor time
(`km.t`); with a scalar or vector time argument the result is the
step-function evaluated at those times, matching R's
`summary(survfit, times = ts)` convention.

Mirrors the `predict(model, type, [t])` surface that `Cox` and
`GeneralHazardModel` expose so downstream code can dispatch on
`AbstractSurvivalModel`-style supertypes uniformly.
"""
StatsAPI.predict(km::KaplanMeier) = predict(km, :survival)

function StatsAPI.predict(km::KaplanMeier, type::Symbol)
    if type === :survival
        return cumprod(1 .- km.∂Λ)
    elseif type === :cumhazard
        return cumsum(km.∂Λ)
    else
        error("Unsupported predict type `:$type` for KaplanMeier. Supported: `:survival`, `:cumhazard`.")
    end
end

function StatsAPI.predict(km::KaplanMeier, type::Symbol, t::Real)
    if type === :survival
        return km(t)
    elseif type === :cumhazard
        return sum((km.∂Λ[i] for i in eachindex(km.t) if km.t[i] < t); init = 0.0)
    else
        error("Unsupported predict type `:$type` for KaplanMeier. Supported: `:survival`, `:cumhazard`.")
    end
end

StatsAPI.predict(km::KaplanMeier, type::Symbol, ts::AbstractVector) =
    [predict(km, type, t) for t in ts]


"""
    greenwood(S::KaplanMeier, t)

Compute the Greenwood variance estimate for the Kaplan-Meier survival estimator at time `t`.

The Greenwood formula provides an estimate of the variance of the Kaplan-Meier survival function at a given time point. For a fitted Kaplan-Meier object `S`, the variance at time `t` is:

```math
\\widehat{\\mathrm{Var}}[\\hat{S}(t)] = \\hat{S}(t)^2 \\sum_{t_j < t} \\frac{d_j}{Y_j (Y_j - d_j)}
"""
greenwood(S::KaplanMeier, t) = sum(S.∂σ[i] for i in eachindex(S.t) if S.t[i] < t)

"""
    confint(km::KaplanMeier; level::Real=0.95)

Pointwise confidence intervals for the Kaplan-Meier survival function, using the
log-log transform and Greenwood's variance estimate. `level` is the confidence
level (e.g. `0.95` for 95% intervals). Returns a `DataFrame` with columns `time`,
`surv`, `lower`, and `upper`.
"""
function StatsAPI.confint(S::KaplanMeier; level::Real=0.95)
    n = length(S.t)
    surv = ones(Float64, n)
    for i in 1:n
        surv[i] = i == 1 ? 1 - S.∂Λ[1] : surv[i-1] * (1 - S.∂Λ[i])
    end
    # Greenwood variance at each time
    var = zeros(Float64, n)
    acc = 0.0
    for i in 1:n
        acc += S.∂σ[i]
        var[i] = (surv[i]^2) * acc
    end
    # z-value for the two-sided interval at the given confidence level
    z = quantile(Normal(), (1 + level) / 2)
    lower = similar(surv)
    upper = similar(surv)
    for i in 1:n
        if surv[i] == 0.0
            lower[i] = 0.0
            upper[i] = 0.0
        else
            se = sqrt(var[i])
            loglog = log(-log(surv[i]))
            halfwidth = z * se / (surv[i] * abs(log(surv[i])))
            lower[i] = exp(-exp(loglog + halfwidth))
            upper[i] = exp(-exp(loglog - halfwidth))
            # Clamp to [0,1]
            lower[i] = max(0.0, min(1.0, lower[i]))
            upper[i] = max(0.0, min(1.0, upper[i]))
        end
    end
    return DataFrame(time=S.t, surv=surv, lower=lower, upper=upper)
end

# Median survival: the smallest event/censor time at which Ŝ drops to or below
# 0.5, or `NaN` if the curve never reaches 0.5.
function _km_median(km::KaplanMeier)
    surv = cumprod(1 .- km.∂Λ)
    idx = findfirst(<=(0.5), surv)
    return idx === nothing ? NaN : km.t[idx]
end

# Total sample size = number at risk just before the first event/censor time.
_km_nobs(km::KaplanMeier) = isempty(km.Y) ? 0 : km.Y[1]

function Base.show(io::IO, ::MIME"text/plain", km::KaplanMeier)
    med = _km_median(km)
    println(io, "Kaplan-Meier survival estimate")
    println(io, "  n: ", _km_nobs(km), ", events: ", sum(km.∂N))
    println(io, "  median survival: ", isnan(med) ? "NA" : _coef_fmt(med))
end

Base.show(io::IO, km::KaplanMeier) = print(io, "KaplanMeier(n=", _km_nobs(km), ")")