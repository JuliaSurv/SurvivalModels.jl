struct CoxV0 <: Cox
    X::Matrix{Float64}
    T::Vector{Float64}
    Δ::Vector{Int64}
    function CoxV0(T,Δ,X)
        o = sortperm(T)
        new(X[o,:],T[o],Δ[o])
    end
end

function getβ(M::CoxV0)
    B0 = zeros(nvar(M))
    o = optimize(par -> cox_nllh(par, M.T, M.Δ, M.X), B0, method=NelderMead, iterations=1000)
    return o.minimizer 
end

function cox_nllh(β, t, δ, X)
    Xβ = X * β # linear predictor.
    θ = exp.(Xβ)

    # we could just sum them for times that are equals. 
    # this maping could be done once. 

    prev_i = firstindex(t) # first index of equal times.
    llh = zero(eltype(β))
    for i in eachindex(t)
        # update the first index of equal times: 
        if t[prev_i] < t[i]
            prev_i = i
        end
        if δ[i]
            llh += - Xβ[i] + log(sum(@view θ[prev_i:end]))
        end
    end
    return llh
end