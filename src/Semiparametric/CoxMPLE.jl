function mlogplik(par, times, status, des, times_obs, nobs)
    x_beta = des * par
    val = -sum(x_beta[status])
    for i in 1:nobs
        rs = findall(times .>= times_obs[i]) # risk set.
        # rs = i:nobs # ? Not working because of ties. 
        val += logsumexp(x_beta[rs])
    end
    return val
end


"""
    CoxModel(init, times, status, des, method, maxit)
    fit(CoxModel, @formula(Surv(T,Δ)~predictors), dataset, method, maxit)

Maximum Partial likelihood estimation in the semiparametric 
Cox Proportional Hazards model. W recomand using the formula interface. 

- times : times to event
- status: vital status indicators (true or 1 = observed, false or 0 = censored)
- des: design matrix for hazard-level effects

- method: An algorithm from `Optim.jl`, e.g. `NelderMead`, `Newton`, `LBFGS`, `ConjugateGradient` or  `GradientDescent`. For the moment our implementation only supports Optim.jl, but a switch to Optimizations.jl will be done in a second step.

maxit: maximum number of iterations in "method"
"""
struct CoxModel
    par::Vector{Float64}
    times::Vector{Float64}
    status::Vector{Bool}
    des::Matrix{Float64}
    times_obs::Vector{Float64}
    nobs::Int64
    function CoxModel(init, times, status, des, method, maxit)
        o = sortperm(times)
        times = times[o]
        status = status[o]
        des = des[o,:]

        status = Bool.(status)
        nobs = sum(status)
        times_obs = times[status]
        optimiser = optimize(par -> mlogplik(par, times, status, des, times_obs, nobs), init, method=method, iterations=maxit)
        return new(
            optimiser.minimizer,
            times,
            status,
            des,
            times_obs,
            nobs
        )
    end
end

mlogplik(X::CoxModel) = mlogplik(X.par, X.times, X.status, X.des, X.times_obs, X.nobs)

function StatsBase.fit(::Type{CoxModel},formula::FormulaTerm, df::DataFrame, method, maxit)
    formula_applied = apply_schema(formula,schema(df))
    predictors = modelcols(formula_applied.rhs, df)
    resp = modelcols(formula_applied.lhs, df)
    return CoxModel(fill(0.0, size(predictors,2)), resp[:,1], resp[:,2], predictors, method, maxit)
end

# Confidence interval : see in HazReg.


# confidence intervals ? 
# pvalues for netsted models ? 
# profile likelyhood ? 