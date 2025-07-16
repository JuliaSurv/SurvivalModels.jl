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

include("Cox/v0.jl")
include("Cox/v1.jl")
include("Cox/v2.jl")
include("Cox/v3.jl")
include("Cox/v4.jl")
include("Cox/v5.jl")

# Calculate the hessian matrix for the different Cox versions:

# Hessian for CoxV0, CoxV1, CoxV4
function getX(M::Union{CoxV0,CoxV1,CoxV4})
    return M.X
end
function getX(M::CoxV5)
    return M.Xᵗ'
end
function get_hessian(M::T, β::Vector{Float64}) where T <: Union{CoxV0, CoxV1, CoxV4, CoxV5}
    n, m = nobs(M), nvar(M)
    X = getX(M)
    η = X * β
    r = exp.(η)
    H = zeros(m, m)
    for i in 1:n
        if M.Δ[i] == 1
            risk_indices = i:n 
            S0_Ri = 0.0
            S1_Ri = zeros(m)
            S2_Ri = zeros(m, m)

            for j in risk_indices
                r_j = r[j]
                X_j = @view X[j, :]
                
                S0_Ri += r_j
                S1_Ri .+= r_j .* X_j
                S2_Ri .+= r_j .* (X_j * X_j')
            end

            invS0_Ri = 1.0 / S0_Ri
            for k in 1:m
                for l in 1:m
                    term1 = S2_Ri[k, l] * invS0_Ri
                    term2 = (S1_Ri[k] * S1_Ri[l]) * (invS0_Ri^2)
                    H[k, l] += (term1 - term2)
                end
            end
        end
    end
    return H
end

function get_H(M::Union{CoxV0, CoxV1, CoxV4, CoxV5}, β::Vector{Float64})
    return get_hessian(M, β)
end

# For CoxV2 and CoxV3 we already use the hessian: 
function get_H(M::CoxV2, β::Vector{Float64})
    _, hess = deriv_loss(β, M)
    return hess
end

function get_H(M::CoxV3, ::Vector{Float64})
    return M.H 
end

function StatsBase.fit(::Type{T}, formula::FormulaTerm, df::DataFrame) where T<:Cox
    CoxVersion = isconcretetype(T) ? T : CoxV3
    formula_applied = apply_schema(formula, schema(df))

    resp = modelcols(formula_applied.lhs, df)
    X = modelcols(formula_applied.rhs, df)
    time = resp[:, 1]
    status = Bool.(resp[:, 2])
    model = CoxVersion(time, status, X)

    # Get the return output:
    # β coefficients:
    beta = getβ(model)
    # Standard Error:
    H_matrice = get_H(model, beta)
    vcov_matrix = inv(H_matrice)
    se = sqrt.(diag(vcov_matrix))
    #Z-Score:
    z_scores = similar(beta, Float64)   
    for i in eachindex(beta)
        z_scores[i] = beta[i] / se[i]
    end
    #P-values: 
    p_values = 2 .* ccdf.(Normal(), abs.(z_scores))

    predictor_names = coefnames(formula_applied.rhs)  
    return DataFrame(
        Predictor = predictor_names,
        β = beta,
        SE = se,
        P_Value = p_values,
        z = z_scores 
    )
end


