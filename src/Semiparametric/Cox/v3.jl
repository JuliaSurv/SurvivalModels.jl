"""
The third implementation of the Cox proportional hazards model represents a highly optimized and significantly 
faster iteration compared to previous implementation, CoxV2.

- Xᵗ::Matrix{Float64}: The design matrix of covariates, transposed (m rows, n columns)
- sX::Vector{Float64}: Sum of X' multiplied by Δ
- T::Vector{Float64}: The observed times sorted in descending order.
- Δ::Vector{Bool}: The event indicator vector (true for event, false for censoring)
- loss::Vector{Float64}: Stores the current negative partial log-likelihood value
- G::Vector{Float64}: Stores the gradient vector  
- H::Matrix{Float64}: Stores the Hessian matrix 
- S₁::Vector{Float64}: Sum of rⱼxₖⱼ
- S₂::Matrix{Float64}: Sum of rⱼxₖⱼ * xⱼ
- μ::Vector{Float64}: Updates the gradient and Hessian
- η::Vector{Float64}: ηi = Xiβ
- r::Vector{Float64}: ri = exp(ηi)
- R::Vector{UnitRange{Int64}}:
"""

struct CoxV3 <: CoxLLH
    Xᵗ::Matrix{Float64}        # shape: (m, n)
    sX::Vector{Float64}       # shape: (m)
    T::Vector{Float64}        # shape: (n)
    Δ::Vector{Bool}           # shape: (n)
    loss::Vector{Float64}     # scalar wrapped in a vector
    G::Vector{Float64}        # shape: (m)
    H::Matrix{Float64}        # shape: (m, m)
    S₁::Vector{Float64}       # shape: (m)
    S₂::Matrix{Float64}       # shape: (m, m)
    μ::Vector{Float64}        # shape: (m)
    η::Vector{Float64}        # shape: (n)
    r::Vector{Float64}        # shape: (n)
    R::Vector{UnitRange{Int64}}
    function CoxV3(T, Δ, X)
        o = reverse(sortperm(T))
        n, m = size(X)
        sX = X' * Δ
        loss     = zeros(1)
        η, r     = zeros(n), zeros(n)
        G, S₁, μ = zeros(m), zeros(m), zeros(m)
        H, S₂    = zeros(m, m), zeros(m, m)

        To = T[o]
        Xoᵗ = X[o,:]'
        Δo = Bool.(Δ[o])
        R = Vector{UnitRange{Int64}}()
        j = 1
        while j <= n
            j₀ = j
            t = To[j]
            while true
                j += 1
                if j > n || To[j] != t
                    break
                end
            end
            push!(R, j₀:(j-1))
        end
        new(Xoᵗ, sX, To, Δo, loss, G, H, S₁, S₂, μ, η, r, R)
    end
end 
nobs(M::CoxV3) = size(M.Xᵗ,2) # X is stored transposed 
nvar(M::CoxV3) = size(M.Xᵗ,1) # X is stored transposed
function update!(β, M::CoxV3)
    M.G .= M.sX
    fill!(M.H, 0)
    fill!(M.S₁, 0)
    fill!(M.S₂, 0)
    M.loss[1] = 0.0

    m = nvar(M)
    mul!(M.η, M.Xᵗ', β)
    @inbounds @simd for j in eachindex(M.r, M.η)
        M.r[j] = exp(M.η[j])
    end

    S₀, j = 0.0, 1
    @inbounds for riskrange in M.R
        sδη, nδ = 0.0, 0
        for j in riskrange
            rⱼ, δⱼ, ηⱼ = M.r[j], M.Δ[j], M.η[j]
            S₀  += rⱼ
            sδη += δⱼ * ηⱼ
            nδ  += δⱼ
            xⱼ = view(M.Xᵗ, : , j)
            for k in 1:m
                rⱼxₖⱼ = rⱼ * xⱼ[k]
                M.S₁[k] += rⱼxₖⱼ
                for l in 1:k
                    M.S₂[l, k] += rⱼxₖⱼ * xⱼ[l]
                end
            end
        end
        if nδ>0
            M.loss[1] += sδη - nδ * log(S₀) 
            invS₀ = 1 / S₀
            for k in 1:m
                μₖ = M.S₁[k] * invS₀
                M.μ[k] = μₖ
                M.G[k] -= nδ * μₖ
                for l in 1:k
                    M.H[l, k] += nδ * (M.S₂[l, k] * invS₀ - μₖ * M.μ[l])
                end
            end
        end
    end

    @inbounds for k in 1:m, l in 1:(k - 1)
        M.H[k, l] = M.H[l, k]
    end

    β .+= M.H \ M.G
    return nothing
end
