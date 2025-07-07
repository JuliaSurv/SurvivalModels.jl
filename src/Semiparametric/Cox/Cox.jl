"""
    StatsBase.fit(Cox, @formula(Surv(T,Δ)~predictors), dataset)

    arguments: 
    - T: The Cox model type to fit (CoxV3)
    - formula: A StatsModels.FormulaTerm specifying the survival model
    - df: A DataFrame containing the variables specified in the formula

    returns: 
    - predictor: A Vector{String} containing the names of the predictor variables included in the model
    - beta: A Vector{Float64} containing the estimated regression coefficients (β​) for each predictor
    - se: A Vector{Float64} containing the standard errors of the estimated regression coefficients
    - loglikelihood: A Vector{Float64} containing the log-likelihood of the fitted model. This value is repeated for each predictor row 

    - coef: A vector of the estimated coefficients
    - formula: The applied formula

Example: 
ovarian = dataset("survival", "ovarian")
ovarian.FUTime = Float64.(ovarian.FUTime) (Time column needs to be Float64 type)
ovarian.FUStat = Bool.(ovarian.FUStat) (Status column needs to be Bool type)
model = fit(Cox, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)


Types: 
- Cox : the base abstract type
- CoxGrad<:Cox : abstract type for Cox models that are solved using gradient-based optimization
- CoxLLH<:CoxGrad : abstract type for Cox models that are solved by optimizing the log-likelihood

"""
abstract type Cox end
nobs(M::Cox) = size(M.X,1) # Default to X being (n,m), should redefine for other choices; 
nvar(M::Cox) = size(M.X,2)
function loss(beta, M::Cox)

    # Requires the presence of 
    # M.X
    # M.Δ
    # M.T
    # This is not very efficient. 
    η = M.X*beta
    return dot(M.Δ, log.((M.T .<= M.T') * exp.(η)) .- η)
end

abstract type CoxGrad<:Cox end
abstract type CoxLLH<:CoxGrad end

function getβ(M::CoxGrad; max_iter = 10000, tol = 1e-9)

    # Requires the presence of: 
    # update!(β, M)
    # i.e all models exept the v1 i think.
    β = zeros(nvar(M))
    βᵢ = similar(β)
    for i in 1:max_iter
        βᵢ .= β
        update!(β, M) 
        gap = L2dist(βᵢ, β)
        if gap < tol
            break
        end
    end
    return β
end

function getβ(M::CoxLLH; max_iter = 10000, tol = 1e-9)
    
    β = zeros(nvar(M))
    llh_prev = llh_new = M.loss[1]
    for i in 1:max_iter
        llh_prev = llh_new
        update!(β, M) 
        llh_new = M.loss[1]
        gap = abs(1 - llh_prev/llh_new)
        if gap < tol
            break
        end
    end
    return β
end

function StatsBase.fit(::Type{T}, formula::FormulaTerm, df::DataFrame) where T<:Cox
    CoxVersion = isconcretetype(T) ? T : CoxV3
    cols = modelcols(formula, df)
    model = CoxVersion(first.(cols[1]), last.(cols[1]), hcat(cols[2]...))
    beta = getβ(model)
    return beta   
    # return DataFrame(
    #     β = beta
    #     se = ???
    #     F = ???
    #     p = ????
    # )
end

