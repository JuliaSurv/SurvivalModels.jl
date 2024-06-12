#= 
=========================================================================
Optimisation function
========================================================================= 
=#

#= CoxMLE function. 

Maximum Partial likelihood estimation in the semiparametric 
Cox Proportional Hazards model


init : initial point for optimisation step for the regression coefficients (beta)

times : times to event

status: vital status indicators (true or 1 = observed, false or 0 = censored)

des: design matrix for hazard-level effects

method: "NM" (NelderMead), "N" (Newton), "LBFGS" (LBFGS), "CG" (ConjugateGradient), "GD" (GradientDescent).  

maxit: maximum number of iterations in "method"
=#

    #= Risk set function =#
function risk_set(t, times)
    # Find indices where times are greater than or equal to t
    return findall(times .>= t)
end

#= - Log-Partial-likelihood =#
function mlogplik(par, times, status, des, times_obs, nobs)
    #= 
    ****************************************************
    Cox Proportional Hazards model 
    ****************************************************
    =#
    #= -Log-likelihood value =#
    x_beta = des * par
    val = -sum(x_beta[status])
    # val2 = 0.0
    for i in 1:nobs
        #= Calculate risk set =#
        rs = risk_set(times_obs[i], times)
        val += logsumexp(x_beta[rs])
    end

    #= return -Log-likelihood value =#
    return val
end
mlogplik(X::CoxModel) = mlogplik(X.par, X.times, X.status, X.des, X.times_obs, X.nobs)


# confidence intervals ? 
# pvalues for netsted models ? 
# profile likelyhood ? 


struct CoxModel{Topt, Tfun}
    par
    times
    status
    des
    times_obs
    nobs
    function CoxModel(init, times, status, des, method, maxit)
        nobs = sum(status)
        times_obs = times[status]
        optimiser = optimize(par -> mlogplik(par, times, status, des, times_obs, nobs), init, method=method, iterations=maxit)
        return new{typeof(optimiser),typeof(mlogplik)}(
            par,
            times,
            status,
            des,
            times_obs,
            nobs
        )
    end
end

function StatsBase.fit(::Type{CoxModel},formula::FormulaTerm, df::DataFrame, method, maxit)
    formula_applied = apply_schema(formula,schema(df))
    predictors = modelcols(formula_applied.rhs, df)
    resp = modelcols(formula_applied.lhs, df)
    return CoxModel(fill(0.0, ncol(predictors)), resp[:,1], resp[:,2], predictors, method, maxit)
end

# Confidence interval : see in HazReg.