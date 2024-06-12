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

function CoxMPLE(; init::Vector{Float64}, times::Vector{Float64}, status::Vector{Bool},
    des::Union{Matrix{Float64},Vector{Float64},Nothing},
    method::String, maxit::Int64)

    nobs = sum(status)
    times_obs = times[status]

    #= Risk set function =#
    function risk_set(t::Float64)
        # Find indices where times are greater than or equal to t
        return findall(times .>= t)
    end


    #= - Log-Partial-likelihood =#
    function mlogplik(par::Vector)
        #= 
        ****************************************************
        Cox Proportional Hazards model 
        ****************************************************
        =#
        #= -Log-likelihood value =#
        x_beta = des * par
        val1 = -sum(x_beta[status])
        val2 = 0
        for i in 1:nobs
            #= Calculate risk set =#
            rs = risk_set(times_obs[i])
            val2 += logsumexp(x_beta[rs])
        end

        #= return -Log-likelihood value =#
        return val1 + val2
    end

    #= Optimisation step =#
    if method == "NM"
        optimiser = optimize(mlogplik, init, method=NelderMead(), iterations=maxit)
    end
    if method == "N"
        optimiser = optimize(mlogplik, init, method=Newton(), iterations=maxit)
    end
    if method == "LBFGS"
        optimiser = optimize(mlogplik, init, method=LBFGS(), iterations=maxit)
    end
    if method == "CG"
        optimiser = optimize(mlogplik, init, method=ConjugateGradient(), iterations=maxit)
    end
    if method == "GD"
        optimiser = optimize(mlogplik, init, method=GradientDescent(), iterations=maxit)
    end

    #= Returns the negative log-likelihood and the optimisation result =#
    return optimiser, mlogplik
end

