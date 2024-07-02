"""
    GHMLE(...)


Maximum likelihood estimation in General Hazards models using 
several parametric baseline hazards

Docs to rewrite....


init : initial point for optimisation step under the parameterisation 
(log(scale), log(shape1), log(shape2), alpha, beta) for scale-shape1-shape2 models or 
(mu, log(scale), alpha, beta) for log-location scale models.

times : times to event

status: vital status indicators (true or 1 = observed, false or 0 = censored)

hstr: hazard structure.  No covariates ("baseline"), AFT model ("AFT"), PH model ("PH"), 
      AH model ("AH"), GH model ("GH").

dist: baseline hazard distribution, on the form of a type <: Distributions.ContinuousUnivariateDistribution. Tested with:
    - LogNormal
    - LogLogistic
    - Weibull
    - Gamma
    - ExponentiatedWeibull
    - GeneralizedGamma
    - PowerGeneralizedWeibull.

des: design matrix for hazard-level effects

des_t: design matrix for time-level effects (it is recommended not to use splines here)

method: one of NelderMead(), Newton(), LBFGS(), ConjugateGradient() or GradientDescent() or any other method taken by Optim.optimize().

maxit: maximum number of iterations of the optimization routine. 

References: 
* [Link to my reference so that people understand what it is](https://myref.com)
"""
function GHMLE(; 
    init::Vector{Float64}, 
    times::Vector{Float64}, 
    status::Vector{Bool},
    hstr::String,
    dist::Type{T},
    des::Union{Matrix{Float64},Vector{Float64},Nothing}, 
    des_t::Union{Matrix{Float64},Vector{Float64},Nothing},
    method::Optim.AbstractOptimizer,
    maxit::Int64) where T<:Distributions.ContinuousUnivariateDistribution

    #= -Log-likelihood =#

    # fix nothings by replacing by empty matrices. 
    des = isnothing(des) ? Array{Float64}(undef,0,0) : des
    des_t = isnothing(des_t) ? Array{Float64}(undef,0,0) : des_t

    # get number of parameters: 
    npd = length(Distributions.params(dist()))
    p = size(des,2)
    q = size(des_t, 2)
    @assert length(init) == npd + p + q # check init vector lenght: 


    # This loglikelyhood function could be completely refactored and simplified using types for the different hazard structures. 
    # it would make a lot of sense. 
    # I did what i could without types, but something better could be done here.. 


    function mloglik(par::Vector)
        # split parameters: 
        d  = dist(exp.(par[1:npd])...) # reconstruct the distribution with these new parameters. 

        if (q == 0) && (p == 0)
            # then only the baseline would be returned since there is no parameters. 
            return general_hazard_llh(d,times,status, 1, 0, 1, 0)
        end

        if p > 0 # then you might want some regression parameters beta. 
            beta = par[(npd+q+1):(npd+q+p)]
            x_beta = des * beta
            exp_beta = exp.(x_beta)
        end

        if q > 0 # then you might want some regression parameters alpha. 
            alpha = q == 1 ? par[npd+1] : par[(npd+1):(npd+q)]
            x_alpha = des_t * alpha
            exp_alpha = exp.(x_alpha)
        end

        if hstr == "PH" # Proportional Hazards models 
            return general_hazard_llh(d,times,status, 1,                    x_beta[status], 1,         exp_beta)
        elseif hstr == "AFT" # Accelerated Failure Time models 
            return general_hazard_llh(d,times,status, exp.(x_beta[status]), x_beta[status], exp_beta,  1)
        elseif hstr == "AH" # Accelerated Hazards models 
            return general_hazard_llh(d,times,status, exp_alpha[status],    0,              exp_alpha, exp.(-x_alpha))
        elseif hstr == "GH" # General Hazards models 
            return general_hazard_llh(d,times,status, exp_alpha[status],    x_beta[status], exp_alpha, exp.(x_beta .- x_alpha))
        end
    end
    optimiser = optimize(mloglik, init, method=method, iterations=maxit)
    return optimiser, mloglik
end
general_hazard_llh(d, T, Δ, α, β, γ, δ) = -sum(loghaz.(d,T[Δ] .* α) .+ β) + sum(cumhaz.(d,T .* γ) .* δ)


#= 
----------------------------------------------------------------------------------------
Function to calculate normal confidence intervals.
It uses the reparameterised log-likelihood function, where all positive parameters are
mapped to the real line using a log-link.
----------------------------------------------------------------------------------------
=#

#= 
FUN   : minus log-likelihood function to be used to calculate the confidence intervals
MLE   : maximum likelihood estimator of the parameters 
level : confidence level
Returns a list containing the upper and lower conf.int limits, the transformed MLE, and std errors
=#


# function ConfInt(; FUN::Function, MLE::Vector{Float64}, level::Float64)
#     dist = Normal(0,1)
#     sd_int = abs(quantile(dist, 0.5*(1-level)))

#     HESS = ForwardDiff.hessian(FUN, MLE)
#     Fisher_Info = inv(HESS)
#     Sigma = sqrt.(diag(Fisher_Info))
#     U = MLE .+ sd_int .* Sigma
#     L = MLE .- sd_int .* Sigma
#     CI = hcat(L,U)
    
#     return CI
# end

"""
    simGH(...)


    Details...

Docs to rewrite....


--------------------------------------------------------------------------------------------------
simGH function: Function to simulate times to event from a model with AH, AFT, PH, GH structures
for different parametric baseline hazards.
Distributions: LogNormal, LogLogistic, GenGamma, GGamma, Weibull, PGW, EW.
See: https://github.com/FJRubio67/HazReg
--------------------------------------------------------------------------------------------------
=#

#=
seed  : seed for simulation
n : sample size (number of individuals)
beta  : regression parameters multiplying the hazard for the GH model
        or the regression parameters for AFT and PH models
alpha  : regression parameters multiplying the time scale for the GH model
        or the regression parameters for the AH model
des : Design matrix for the GH model (hazard scale)
      or design matrix for AFT and PH models
des_t : Design matrix for the GH model (time scale)
       or design matrix for the AH model
hstr  : hazard structure (AH, AFT, PH, GH)
dist: distribution passed as an already instanciated distribution from Distributions.jl
baseline  : baseline hazard distribution

Returns a vector containing the simulated times to event

References: 
* [Link to my reference so that people understand what it is](https://myref.com) 
"""
function simGH(; seed::Int64, n::Int64,
    des::Union{Matrix{Float64},Vector{Float64},Nothing}, 
    des_t::Union{Matrix{Float64},Vector{Float64},Nothing},
    alpha::Union{Vector{Float64},Float64,Nothing}, 
    beta::Union{Vector{Float64},Float64,Nothing},
    hstr::String, dist::Distributions.ContinuousDistribution)

    #= Uniform variates =#
    Random.seed!(seed)
    distu = Uniform(0, 1)
    u = rand(distu, n)

    # Also here, these could be simplified a lot by a hazard structure :)
    if hstr == "GH"
        exp_xalpha = exp.(des_t * alpha)
        exp_dif    = exp.(des_t * alpha .- des * beta)
    elseif hstr == "PH"
        exp_xalpha = 1.0
        exp_dif    = exp.(-des * beta)
    elseif hstr == "AFT"
        exp_xalpha = exp.(des * beta)
        exp_dif    = 1.0
    elseif hstr == "AH"
        exp_xalpha = exp.(des_t * alpha)
        exp_dif    = exp.(des_t * alpha)
    end

    # Simulating the times to event
    p0 = 1 .- exp.(log.(1 .- u) .* exp_dif)
    times = quantile.(dist,p0) ./ exp_xalpha

    return times
end