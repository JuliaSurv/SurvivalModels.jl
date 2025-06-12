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

const design = Dict(
    # Label => (constructor, plotting color)
    #"R" => (CoxVR, :red),
    "Jl"=> (CoxVJ, :blue),
    "V1"=> (CoxV1, :orange),
    "V2"=> (CoxV2, :brown),
    "V3"=> (CoxV3, :purple),
    "V4"=> (CoxV4, :green),
    "V5"=> (CoxV5, :black)
);

    
function StatsBase.fit(::Type{Cox}, formula::FormulaTerm, version::String = "V3")
    cox_version = get(design, version)
    version = cox_version[1]
    formula_applied = apply_schema(formula,schema(df))
    resp = modelcols(formula_applied.lhs, df)
    X = modelcols(formula_applied.rhs, df)
    time = resp[:, 1]
    status = Bool.(resp[:, 2])
    model = version(time, status, X)

    beta = getβ(model)
    return (model=model, coef=beta, formula=formula_applied)   
end

