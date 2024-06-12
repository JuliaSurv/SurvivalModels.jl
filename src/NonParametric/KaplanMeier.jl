struct KaplanMeier{TVt,TVd,TVn}
    t::TVt
    d::TVd
    n::TVn
    ∂Λ::TVt
    ∂σ::TVt
    function KaplanMeier(Y::VTY,Δ::VTΔ) where {VTY<:AbstractVector, VTΔ<:AbstractVector}
        @assert length(Y) == length(Δ)
        o = sortperm(Y)
        strY = Y[o]
        strΔbool = Bool.(Δ)[o]
        t = unique(strY)
        d = zeros(Int,length(t))
        n = zeros(Int,length(t))
        ∂Λ = zeros(eltype(Y),length(t))
        ∂σ = zeros(eltype(Y),length(t))
        for i in eachindex(t)
            # compute number of observed events and number of persons at risk: 
            for j in eachindex(strY)
                d[i] += (strY[j] == t[i]) && !(strΔbool[j])
                n[i] += strY[j] >= t[i]
            end
            ∂Λ[i] = d[i] / n[i]
            ∂σ[i] = d[i] / (n[i] * (n[i] - d[i]))
        end
        return new{typeof(t),typeof(d),typeof(n)}(t,d,n)
    end
end
(S::KaplanMeier)(t) = prod(1 - S.∂Λ[i] for i in eachindex(S.t) if S.t[i] < t; init= one(t))
greenwood(S,t) = sum(S.∂σ[i] for i in eachindex(S.t) if S.t[i] < t; init=zero(t))

function StatsAPI.confint(S::KaplanMeier; level::Real=0.05)
    # TODO 
end
# Other methods to implement for this ? 
# Maybe a vignette at least ? 

