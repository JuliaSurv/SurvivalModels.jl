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
    lhs_vars = StatsModels.termvars(formula.lhs)
    resp = modelcols(formula.lhs, df[:, lhs_vars])
    Tvec = getindex.(resp, 1)
    Δvec = getindex.(resp, 2)
    return KaplanMeier(Tvec, Δvec)
end

# Survival estimate Ŝ(t)
(S::KaplanMeier)(t) = prod((1 - S.∂Λ[i] for i in eachindex(S.t) if S.t[i] < t); init=1.0)

"""
    greenwood(S::KaplanMeier, t)

Compute the Greenwood variance estimate for the Kaplan-Meier survival estimator at time `t`.

The Greenwood formula provides an estimate of the variance of the Kaplan-Meier survival function at a given time point. For a fitted Kaplan-Meier object `S`, the variance at time `t` is:

```math
\\widehat{\\mathrm{Var}}[\\hat{S}(t)] = \\hat{S}(t)^2 \\sum_{t_j < t} \\frac{d_j}{Y_j (Y_j - d_j)}
"""
greenwood(S::KaplanMeier, t) = sum(S.∂σ[i] for i in eachindex(S.t) if S.t[i] < t)

function StatsAPI.confint(S::KaplanMeier; level::Real=0.05)
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
    # z-value for confidence level
    z = quantile(Normal(), 1 - level/2)
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