"""
    StatsBase.fit(Cox, @formula(Surv(T,Δ)~predictors), dataset)

Arguments: 
- T: The Cox model type to fit (CoxDefault)
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

We need to add details about the different prediction types here. 

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

include("Cox/CoxNM.jl")
include("Cox/coxOptim.jl")
include("Cox/CoxHessian.jl")
include("Cox/CoxDefault.jl")
include("Cox/CoxApprox.jl")


# Extract the matrix of X's : 
getX(M::CoxMethod) = M.X
getX(M::CoxDefault) = M.Xᵗ'
nobs(M::CoxMethod) = size(getX(M),1) # Default to X being (n,m)
nvar(M::CoxMethod) = size(getX(M),2)
get_perm(M::CoxMethod) = M.o
get_og_X(M::CoxMethod) = getX(M)[sortperm(get_perm(M)), :]

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
get_hessian(M::CoxHessian, β) = deriv_loss(β, M)[2]
get_hessian(M::CoxDefault, _) = M.H
get_hessian(C::Cox) = get_hessian(C.M, C.β)

# Compute Harrel's C-index: 
harrells_c(C::Cox) = harrells_c(C.M.T, C.M.Δ, getX(C.M) * C.β)

function predict_lp(C::Cox; centered::Bool=true, og::Bool=true)
    X = og ? get_og_X(C.M) : getX(C.M)
    η = X * C.β 

    centered || return η

    d = nvar(C.M) 
    mX = zeros(1, d)
    cat = C.pred_types .== :categorical
    for i in 1:d
        mX[1,i] = cat[i] ? mode(sort(X[:,i])) : mean(X[:,i])
    end
    return η .- (mX * C.β)[1]
end
function baseline_hazard(C::Cox; centered::Bool = false, og::Bool=false)
    η = predict_lp(C, centered = centered, og = og)
    R = exp.(η)
    t_j = sort(unique(C.M.T))
    H0 = 0.0 
    hazard = zero(t_j)
    for (j, tⱼ) in enumerate(t_j)
        d_j = length(findall((C.M.T .== tⱼ) .& (C.M.Δ .== true)))
        R_j = findall(C.M.T .>= tⱼ)
        sum_den = sum(R[R_j])
    
        if d_j > 0 && sum_den > 0
            H_0 = d_j / sum_den
            H0 += H_0 
        end
        hazard[j] = H0
    end
    return hazard
end
function predict_terms(C::Cox, og=true)
    X = og ? get_og_X(C.M) : getX(C.M)
    β = C.β          
    d = nvar(C.M) 

    rez = X .* β'
    cat = C.pred_types .== :categorical
    for i in 1:d
        rez[:,i] .-= (cat[i] ? mode(sort(X[:,i])) : mean(X[:,i])) * β[i]
    end
    return rez
end
predict_expected(C::Cox) = baseline_hazard(C, centered=true) .* exp.(predict_lp(C))
predict_survival(C::Cox) = exp.(-predict_expected(C))
predict_risk(C::Cox; type = :risk) = exp.(predict_lp(C))
function predict(C::Cox, type::Symbol=:lp)
    type==:lp && return predict_lp(C)
    type==:risk && return predict_risk(C)
    type==:expected && return predict_expected(C)
    type==:terms && return predict_terms(C)
    type==:survival && return predict_survival(C)
    error("The prediction you want was not understood. Please pass a `type` parameter among (:lp, :risk, :expected, :terms, :survival). See `?Cox` for details.")
end

function StatsBase.fit(::Type{T}, formula::FormulaTerm, df::DataFrame) where T<:Union{CoxMethod,Cox}
    CoxWorkerType = (isconcretetype(T) || T != Cox) ? T : CoxDefault
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
