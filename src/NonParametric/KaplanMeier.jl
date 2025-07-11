"""
    KaplanMeier(T, Δ)

Efficient Kaplan-Meier estimator.

Arguments:
- `T`: Vector of event or censoring times.
- `Δ`: Event indicator vector (`1` if event, `0` if censored).

Stores:
- `t`: Sorted unique event times.
- `∂N`: Number of uncensored death at each time point.
- `Y`: Number of at risk individual at each time point.
- `∂Λ`: Increments of cumulative hazard.
- `∂σ`: Greenwood variance increments.
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
        To = T[o]
        Δo = Bool.(Δ[o])
        t = unique(To).*1.0 # Convert to Float64 for consistency
        N, n = length(To), length(t)
        ∂N, Y, ∂Λ, ∂σ = zeros(Int, n), zeros(Int, n), zero(t), zero(t)
        j = 1
        at_risk = N
        for i in 1:n
            ti = t[i]
            # Compute size and n_events of the risk set:
            rs_size, rs_events = 0, 0
            while To[j] == ti && j <= N 
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
        new{eltype(T)}(t, ∂N, Y, ∂Λ, ∂σ)
    end
end

function StatsBase.fit(::Type{T}, formula::FormulaTerm, df::DataFrame) where {T<:KaplanMeier}
    resp = modelcols(apply_schema(formula,schema(df)).lhs,df)
    return KaplanMeier(resp[:,1], resp[:,2])
end

# Survival estimate Ŝ(t)
(S::KaplanMeier)(t) = prod(1 - S.∂Λ[i] for i in eachindex(S.t) if S.t[i] < t)

# Greenwood variance estimate at time t
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