"""
    StatsBase.fit(CoxV3, @formula(Surv(T,Δ)~predictors), dataset)

    arguments: 
    - T: The Cox model type to fit (CoxV3)
    - formula: A StatsModels.FormulaTerm specifying the survival model
    - df: A DataFrame containing the variables specified in the formula

    returns: 
    - model: The fitted Cox model
    - coef: A vector of the estimated coefficients
    - formula: The applied formula

Example: 
 ovarian = dataset("survival","ovarian")
formula = @formula(FUTime + FUStat ~ Age + ECOG_PS)
results = StatsBase.fit(CoxV3, formula, ovarian)


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
    formula_applied = apply_schema(formula,schema(df))
    resp = modelcols(formula_applied.lhs, df)
    X = modelcols(formula_applied.rhs, df)
    time = resp[1]
    status = Bool.(resp[2])
    model = CoxVersion(time, status, X)
    beta = getβ(model)
    return (model=model, coef=beta, formula=formula_applied)   
end

