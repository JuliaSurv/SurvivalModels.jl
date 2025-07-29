"""
    StatsBase.fit(Cox, @formula(Surv(T,Δ)~predictors), dataset)

Arguments: 
- T: The Cox model type to fit (CoxV3)
- formula: A StatsModels.FormulaTerm specifying the survival model
- df: A DataFrame containing the variables specified in the formula

Returns: 
- predictor: A Vector{String} containing the names of the predictor variables included in the model
- beta: A Vector{Float64} containing the estimated regression coefficients (β​) for each predictor
- se: A Vector{Float64} containing the standard errors of the estimated regression coefficients
- loglikelihood: A Vector{Float64} containing the log-likelihood of the fitted model. This value is repeated for each predictor row 

- coef: A vector of the estimated coefficients
- formula: The applied formula

Example:

```julia
ovarian = dataset("survival", "ovarian")
ovarian.FUTime = Float64.(ovarian.FUTime) (Time column needs to be Float64 type)
ovarian.FUStat = Bool.(ovarian.FUStat) (Status column needs to be Bool type)
model = fit(Cox, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)
```

Types: 
- Cox : the base abstract type
- CoxGrad<:Cox : abstract type for Cox models that are solved using gradient-based optimization
- CoxLLH<:CoxGrad : abstract type for Cox models that are solved by optimizing the log-likelihood

"""
abstract type CoxMethod end
abstract type CoxGrad<:CoxMethod end
abstract type CoxLLH<:CoxGrad end
struct Cox{CM}
    M::CM
    β::Vector{Float64}
    pred_names::Vector{Symbol}
    pred_types::Vector{Symbol}
    function Cox(obj::CM, names, types) where {CM <: CoxMethod}
        new{CM}(obj, getβ(obj), names, types)
    end
end

nobs(M::CoxMethod) = size(M.X,1) # Default to X being (n,m)
nvar(M::CoxMethod) = size(M.X,2)
function loss(beta, M::CoxMethod)
    η = M.X*beta
    return dot(M.Δ, log.((M.T .<= M.T') * exp.(η)) .- η)
end

function getβ(M::CoxGrad; max_iter = 10000, tol = 1e-9)
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


# Extract the matrix of X's : 
getX(M::CoxMethod) = M.X
getX(M::Union{CoxV3,CoxV5}) = M.Xᵗ'

# Extract the hessian: 
function get_hessian(M::CoxMethod, β)
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
get_hessian(M::CoxV2, β) = deriv_loss(β, M)[2]
get_hessian(M::CoxV3, _) = M.H
get_hessian(C::Cox) = get_hessian(C.M, C.β)

# Compute Harrel's C-index: 
harrells_c(C::Cox) = harrells_c(C.M.T, C.M.Δ, getX(C.M) * C.β)

function get_og_X(M::CoxV3)
    return M.X_og
end

function baseline_hazard(C::Cox; centered::Bool = false)
    #X = get_og_X(C.M)
    X = getX(C.M)
    η = X * C.β # (LP indv)
    R = exp.(η)
    d = nvar(C.M)
    
    if centered
        mean_X = zeros(1, d)
        for i in 1:d
            if C.pred_types[i] == :categorical
                # mean_X[1,i] = mode(X[:,i])
                mean_X[1,i] = 0.0
            else
                mean_X[1,i] = mean(X[:,i])
            end
        end
        LP_sample = mean_X * C.β
        LP_c = η .- LP_sample[1]
        R = exp.(LP_c)
    end

    t_j = sort(unique(C.M.T))

    H0 = 0.0 
    time = Float64[]
    hazard = Float64[]

    for j in t_j
        d_j = length(findall((C.M.T .== j) .& (C.M.Δ .== true)))
        R_j = findall(C.M.T .>= j)
        sum_den = sum(R[R_j])
    
        if d_j > 0 && sum_den > 0
            H_0 = d_j / sum_den
            H0 += H_0 
        end
        
        push!(time, j)
        push!(hazard, H0)
    end

    return DataFrame(time = time, hazard = hazard)
end

function predict_lp(C::Cox; type::Symbol= :lp)
    X = get_og_X(C.M)
    η = X * C.β 
    d = nvar(C.M) 

    mean_X = zeros(1, d)
    for i in 1:d
        if C.pred_types[i] == :categorical
            mean_X[1,i] = 0.0 
        else
            mean_X[1,i] = mean(X[:,i])
        end
    end
    LP_sample = (mean_X * C.β)[1]
    
    η = η .- LP_sample
    
    return η
end

function predict_risk(C::Cox; type = :risk)
    return exp.(predict_lp(C))
end 

function predict_expected(C::Cox; type = :expected)
    X = getX(C.M)
    η = X * C.β # (LP indv)
    R = exp.(η)
    d = nvar(C.M)
    

    mean_X = zeros(1, d)
    for i in 1:d
        if C.pred_types[i] == :categorical
            # mean_X[1,i] = mode(X[:,i])
            mean_X[1,i] = 0.0
        else
            mean_X[1,i] = mean(X[:,i])
        end
    end
    LP_sample = mean_X * C.β
    LP_c = η .- LP_sample[1]
    R = exp.(LP_c)

    t_j = sort(unique(C.M.T))

    H0 = 0.0 
    hazard = Float64[]

    for j in t_j
        d_j = length(findall((C.M.T .== j) .& (C.M.Δ .== true)))
        R_j = findall(C.M.T .>= j)
        sum_den = sum(R[R_j])
    
        if d_j > 0 && sum_den > 0
            H_0 = d_j / sum_den
            H0 += H_0 
        end
        
        push!(hazard, H0)
    end

    return hazard .* predict_risk(C)
end

function predict_terms(C::Cox; type = :terms)
    X = get_og_X(C.M)
    β = C.β          
    d = nvar(C.M) 

    res = Matrix{Float64}(undef, nobs(C.M), d)

    for i in 1:d
        terme1 = X[:, i] .*  β[i]
        if C.pred_types[i] == :categorical
            terme2 = 0.0
        else
            terme2 = mean(X[:,i]) * β[i]
        end
        res[:, i] = terme1 .- terme2
    end
    return res
end


function predict_survival(C::Cox; type = :survial)
    return exp.(-predict_expected(C))
end

function StatsBase.fit(::Type{T}, formula::FormulaTerm, df::DataFrame) where T<:Union{CoxMethod,Cox}
    CoxWorkerType = (isconcretetype(T) || T != Cox) ? T : CoxV3
    formula_applied = apply_schema(formula, schema(df))
    resp = modelcols(formula_applied.lhs, df)
    X = modelcols(formula_applied.rhs, df)
    Y, Δ = resp[:, 1], Bool.(resp[:, 2])
    predictor_names = coefnames(formula_applied.rhs)
    catego = typeof.(formula_applied.rhs.terms) .<: CategoricalTerm
    catego = [c ? :categorical : :continuous for c in catego]
    return Cox(CoxWorkerType(Y, Δ, X), Symbol.(predictor_names), catego)
end

function summary(C::Cox)
    # Standard Error, z-scroe, p-values and c-index: 
    se = sqrt.(diag(inv(get_hessian(C.M, C.β))))
    z_scores = C.β ./ se
    p_values = 2 .* ccdf.(Normal(), abs.(z_scores))

    return DataFrame(
        predictor = C.pred_names,
        β = C.β,
        se = se,
        p_values = p_values,
        z_scores = z_scores 
    )
end
function _summary_line(C::Cox)
    c_index = harrells_c(C)
    return "Cox Model (n: $(nobs(C.M)), m: $(nvar(C.M)), method: $(typeof(C).parameters[1]), C-index: $(c_index))"
end

function Base.show(io::IO, C::Cox)
    println(io, _summary_line(C))
    Base.show(summary(C))
end
