"""
    CoxOptim(T, Δ, X)
    fit(CoxOptim, @formula(Surv(T,Δ)~X), data = ...)

The first implementation of the Cox proportional hazards model uses Optim.jl for coefficient estimation.
It uses the BFGS algorithm (with forward-mode automatic differentiation) to minimize the negative partial log-likelihood.

Fields: 
- X::Matrix{Float64}: The design matrix of covariates, where rows correspond to individuals and columns to features.
- T::Vector{Float64}: The observed times, sorted in ascending order
- Δ::Vector{Int64}: The event indicator vector (true for event, false for censoring)
"""
struct CoxOptim<:CoxMethod
    X::Matrix{Float64}
    T::Vector{Float64}
    Δ::Vector{Bool}
    o::Vector{Int64}
    function CoxOptim(T,Δ,X)
        o = sortperm(T)
        new(X[o,:],T[o],Δ[o], o)
    end
end

function getβ(M::CoxOptim)
    B0 = zeros(nvar(M))
    # Minimize the negative partial log-likelihood `loss(β, M)` with Optim's BFGS
    # and a forward-mode AD gradient.
    res = optimize(b -> loss(b, M), B0, BFGS(); autodiff = AutoForwardDiff())
    return res.minimizer
end