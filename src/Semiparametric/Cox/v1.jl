"""
    CoxV1(T, Δ, X)
    fit(CoxV1, @formula(Surv(T,Δ)~X), data = ...)

The first implementation of the Cox proportional hazards model uses optimization libraries (Optimization.jl, Optim.jl) for coefficient estimation.
It uses the BFGS algorithm to minimize the negative partial log-likelihood. 

Fields: 
- X::Matrix{Float64}: The design matrix of covariates, where rows correspond to individuals and columns to features.
- T::Vector{Float64}: The observed times, sorted in ascending order
- Δ::Vector{Int64}: The event indicator vector (true for event, false for censoring)
"""
struct CoxV1<:Cox
    X::Matrix{Float64}
    T::Vector{Float64}
    Δ::Vector{Int64}
    function CoxV1(T,Δ,X)
        o = sortperm(T)
        new(X[o,:],T[o],Δ[o])
    end
end

function getβ(M::CoxV1)
    B0 = zeros(nvar(M))
    f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(f, B0, M)
    sol = solve(prob, Optim.BFGS())
    return sol.u
end