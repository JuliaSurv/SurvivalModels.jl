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

"""
    CoxM(init, times, status, des, method, maxit)
    fit(CoxM, @formula(Surv(T,Δ)~predictors), dataset, method, maxit)

Maximum Partial likelihood estimation in the semiparametric 
Cox Proportional Hazards model. W recomand using the formula interface. 

- times : times to event
- status: vital status indicators (true or 1 = observed, false or 0 = censored)
- des: design matrix for hazard-level effects

- method: An algorithm from `Optim.jl`, e.g. `NelderMead`, `Newton`, `LBFGS`, `ConjugateGradient` or  `GradientDescent`. For the moment our implementation only supports Optim.jl, but a switch to Optimizations.jl will be done in a second step.

maxit: maximum number of iterations in "method"
"""
struct CoxM
    par::Vector{Float64}
    times::Vector{Float64}
    status::Vector{Bool}
    des::Matrix{Float64}
    function CoxM(init, times, status, des, method, maxit)

        o = sortperm(times)
        times = times[o]
        status = Bool.(status[o])
        des = des[o,:]

        # Let me compute here the llh : 
        optimiser = optimize(par -> cox_nllh(par, times, status, des), init, method=method, iterations=maxit)
        return new(
            optimiser.minimizer,
            times,
            status,
            des,
        )
    end
end

mlogplik(X::CoxM) = mlogplik(X.par, X.times, X.status, X.des)

function StatsBase.fit(::Type{CoxM},formula::FormulaTerm, df::DataFrame, method, maxit)
    formula_applied = apply_schema(formula,schema(df))
    predictors = modelcols(formula_applied.rhs, df)
    resp = modelcols(formula_applied.lhs, df)
    return CoxM(fill(0.0, size(predictors,2)), resp[:,1], resp[:,2], predictors, method, maxit)
end

# Confidence interval : see in HazReg.


# confidence intervals ? 
# pvalues for netsted models ? 
# profile likelyhood ? 


# we would like to set this up as an optimization routine from Optimization.jl so that is can be solved using different stuff. 
# we can provide the loss but also the gradient and evne the hessian as formulas are on wikipedia. 



