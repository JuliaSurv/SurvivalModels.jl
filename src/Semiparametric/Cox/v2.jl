"""
    CoxV2(T, Δ, X)
    fit(CoxV2, @formula(Surv(T,Δ)~X), data = ...)

The second implementation of the Cox proportional hazards model uses a Newton-Raphson-like iterative update that directly calculates and utilizes the gradient and Hessian matrix. This version is updating coefficients via the update! function.

Fields:
- X::Matrix{Float64}: The design matrix of covariates, where rows correspond to individuals and columns to features
- T::Vector{Float64}: The observed times, sorted in ascending order
- Δ::Vector{Int64}: The event indicator vector (true for event, false for censoring)
- R::BitMatrix: A boolean risk matrix, where 'R[i,j]' is 'true' if individual 'j' is at risk at time 'T[i]'
"""
struct CoxV2<:CoxGrad
    X::Matrix{Float64}
    T::Vector{Float64}
    Δ::Vector{Bool}
    R::BitMatrix
    function CoxV2(T,Δ,X)
        o = sortperm(T)
        R = T .<= T'
        new(X[o,:],T[o],Δ[o],R[o,o])
    end
end

function deriv_loss(beta, M::CoxV2)
    η = M.X*beta
    eη = exp.(η)
    n,m = nobs(M), nvar(M)

    grad = zeros(m)
    hess = zeros(m,m)
    summ = zeros(m)
    summ2 = zeros(m,m)

    for i in 1:n
        summ .= 0.0
        summ2 .= 0.0
        if M.Δ[i] == 1
            dsum_exp = 0.0
            for j in 1:n
                if M.R[i,j]
                    dsum_exp += eη[j]
                    for k in 1:m
                        summ[k] += eη[j] * M.X[j,k]
                        for l in 1:m
                            summ2[k,l] += eη[j] * M.X[j,k] * M.X[j,l]
                        end
                    end
                end
            end
            for k in 1:m
                grad[k] += M.X[i,k] - (summ[k]/dsum_exp)
                for l in 1:m
                    hess[k,l] += (summ2[k,l] * dsum_exp - summ[k]*summ[l]) / dsum_exp^2
                end
            end
        end
    end
    return -grad, hess
end


function update!(β, M::CoxV2)
    grad, hess = deriv_loss(β, M)
    β .-= hess \ grad  
    return nothing
end