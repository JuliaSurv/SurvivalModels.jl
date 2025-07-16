"""
    CoxV5(T, Δ, X)
    fit(CoxV5, @formula(Surv(T,Δ)~X), data = ...)

The fifth implementation of the Cox proportional hazards model uses a pre-calculated Hessian approximation for faster iterations, like CoxV4. Its approach is similar to CoxV3.

Fields:
- Xᵗ::Matrix{Float64}: The design matrix of covariates, transposed (m rows, n columns)
- sX::Vector{Float64}: Sum of X' multiplied by Δ
- T::Vector{Float64}: The observed times sorted in descending order
- Δ::Vector{Bool}: The event indicator vector (true for event, false for censoring)
- loss::Vector{Float64}: Stores the current negative partial log-likelihood value
- G::Vector{Float64}: Stores the gradient vector.
- S₁::Vector{Float64}:  Sum of rⱼxₖⱼ
- μ::Vector{Float64}: Currently unused in `update!` function
- η::Vector{Float64}: ηi = Xiβ
- r::Vector{Float64}: ri = exp(ηi)
- R::Vector{UnitRange{Int64}}:
- B::Vector{Float64}: Stores the majoration elements of the Hessian matrix
"""
struct CoxV5 <: CoxLLH
    Xᵗ::Matrix{Float64}        # shape: (m, n)
    sX::Vector{Float64}       # shape: (m)
    T::Vector{Float64}        # shape: (n)
    Δ::Vector{Bool}           # shape: (n)
    loss::Vector{Float64}     # scalar wrapped in a vector
    G::Vector{Float64}        # shape: (m)
    S₁::Vector{Float64}       # shape: (m)
    μ::Vector{Float64}        # shape: (m)
    η::Vector{Float64}        # shape: (n)
    r::Vector{Float64}        # shape: (n)
    R::Vector{UnitRange{Int64}}
    B::Vector{Float64}
    function CoxV5(T, Δ, X)
        o = reverse(sortperm(T))
        n, m = size(X)
        sX = X' * Δ
        loss     = zeros(1)
        η, r     = zeros(n), zeros(n)
        G, S₁, μ, B = zeros(m), zeros(m), zeros(m), zeros(m)

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

        Ro = StatsBase.competerank(To)
        I = Ro[Δo]

        J = zeros(Int64,n)
        for rⱼ in I
            J[rⱼ] += 1
        end
        for j in 2:n
            J[j] += J[j-1]
        end


        K = Int64[]
        Jₖ₋₁ = 0
        for Jₖ in J
            push!(K,length((Jₖ₋₁+1):Jₖ))
            Jₖ₋₁ = Jₖ
        end
        
        # Compute hessian bounds:
        for l in 1:m # for each dimension. 
            lastj = n+1
            Mₓ = Xoᵗ[l,end]
            mₓ = Xoᵗ[l,end]
            Bₗ = 0.0
            for i in n:-1:1
                for j in Ro[i]:(lastj-1)
                    xⱼ = Xoᵗ[l,j]
                    Mₓ = max(Mₓ, xⱼ)
                    mₓ = min(mₓ, xⱼ)
                end
                lastj = Ro[i]
                Bₗ += (1/4) * Δo[i] * (Mₓ-mₓ)^2
            end
            B[l] = Bₗ
        end
        new(Xoᵗ, sX, To, Δo, loss, G, S₁, μ, η, r, R, B)
    end
end 
nobs(M::CoxV5) = size(M.Xᵗ,2) # X is stored transposed 
nvar(M::CoxV5) = size(M.Xᵗ,1) # X is stored transposed
function update!(β, M::CoxV5)
    M.G .= M.sX
    fill!(M.S₁, 0)
    M.loss[1] = 0.0
    m = nvar(M)
    mul!(M.η, M.Xᵗ', β)
    S₀, j, L = 0.0, 1, 0.0
    @inbounds for riskrange in M.R
        nδ = 0
        for j in riskrange
            δⱼ, ηⱼ = M.Δ[j], M.η[j]
            rⱼ = exp(ηⱼ)
            S₀  += rⱼ
            L += δⱼ * ηⱼ
            nδ  += δⱼ
            for k in 1:m
                M.S₁[k] += rⱼ * M.Xᵗ[k,j]
            end
        end
        if nδ>0
            L -= nδ * log(S₀)
            normalizer = nδ / S₀
            for k in 1:m
                M.G[k] -= normalizer * M.S₁[k]
            end
        end
    end
    M.loss[1] = L
    β .+= M.G ./ M.B
    return nothing
end
