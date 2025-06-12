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