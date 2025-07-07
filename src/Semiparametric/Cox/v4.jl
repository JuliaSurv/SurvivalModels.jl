"""
    CoxV4(T, Δ, X)
    fit(CoxV4, @formula(Surv(T,Δ)~X), data = ...)

The fourth implementation of the Cox proportional hazards model uses Hessian approximation based on a pre-calculated estimation. This version was created for when it might be difficult to work with full Hessian , offering faster iterations by using a Hessian approximation.

Fields:
    - X::Matrix{Float64}: The design matrix of covariates, where rows correspond to individuals and columns to features
    - T::Vector{Float64}: The observed times sorted in ascending order
    - Δ::Vector{Bool}: The event indicator vector (true for event, false for censoring)
    - sX::Vector{Float64}: Sum of X' multiplied by Δ
    - G::Vector{Float64}: Stores the gradient vector
    - η::Vector{Float64}: ηi = Xiβ
    - A::Vector{Float64}: Ai = exp(ηi)
    - B::Vector{Float64}: Stores the majoration elements of the Hessian matrix
    - C::Vector{Float64}: Used in the mkA! function
    - K::Vector{Int64}: Number of events at each unique observed event time
    - loss::Vector{Float64}: Stores the current negative partial log-likelihood value, used in CoxLLH getβ
"""
struct CoxV4<:CoxLLH
    X::Matrix{Float64}
    T::Vector{Float64}
    Δ::Vector{Bool}
    sX::Vector{Float64}
    G::Vector{Float64}
    η::Vector{Float64}
    A::Vector{Float64}
    B::Vector{Float64}
    C::Vector{Float64}
    K::Vector{Int64}
    loss::Vector{Float64}
    function CoxV4(T,Δ,X)

        # Allocate: 
        n,m = size(X)
        G, B  = zeros(m), zeros(m)
        η, A, C = zeros(n), zeros(n), zeros(n)

        # Precompute a few things: 
        # This should also be optimized, its taking a lot of time right now..
        o = sortperm(T)
        To = T[o]
        Δo = Bool.(Δ[o])
        Xo = X[o,:]
        sX = X'Δ
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
            Mₓ = Xo[end,l]
            mₓ = Xo[end,l]
            Bₗ = 0.0
            for i in n:-1:1
                for j in Ro[i]:(lastj-1)
                    Mₓ = max(Xo[j,l], Mₓ)
                    mₓ = min(Xo[j,l], mₓ)
                end
                lastj = Ro[i]
                Bₗ += (1/4) * Δo[i] * (Mₓ-mₓ)^2
            end
            B[l] = Bₗ
        end

        # Instantiate: 
        new(Xo, To, Δo, sX, G, η, A, B, C, K, [1.0])
    end
end

function mkA!(A, C, K)
    # This is equivalent to : 
      C .= reverse(cumsum(reverse(A)))
      A .= A .* cumsum(K ./ C)
      return sum(K .* log.(C))
    # But 40% faster by fusing loop as much as possible.
    # We exploited the trick that 
    #   reverse(cumsum(reverse(X))) == sum(X) .- cumsum([0,X[1:end-1]...]) 
    
    # s, c = sum(A), 0.0
    # L = 0.0
    # for (i, (aᵢ, kᵢ)) in enumerate(zip(A, K))
    #     if kᵢ > 0
    #         c += kᵢ / s
    #         L += kᵢ * log(s)
    #     end
    #     s -= aᵢ 
    #     A[i] *= c
    # end 
    # return L
end
function update!(β, M::CoxV4)
    mul!(M.η, M.X, β)    # O(nm)
    M.A .= exp.(M.η)     # O(n)
    L = mkA!(M.A, M.C, M.K)   # O(n)
    M.loss[1] = dot(M.Δ, M.η) - L # O(n)
    mul!(M.G, M.X', M.A) # O(nm)
    M.G .= M.sX .- M.G   # O(m)
    β .+= M.G ./ M.B     # O(m)
    return nothing
end