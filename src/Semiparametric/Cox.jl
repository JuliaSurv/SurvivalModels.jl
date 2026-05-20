"""
    StatsBase.fit(Cox, @formula(Surv(T,Δ)~predictors), dataset)

Arguments: 
- Cox: The model to fit, e.g. Cox. Could be specified to any of `CoxNM, CoxOptim, CoxApprox` or `CoxDefault` if you want different sovers to be used, see their own documentations. Default is `CoxDefault`.
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
    formula::Union{FormulaTerm, Nothing}
    function Cox(obj::CM, names, types; formula::Union{FormulaTerm, Nothing}=nothing) where {CM <: CoxMethod}
        new{CM}(obj, getβ(obj), names, types, formula)
    end
end

function loss(beta, M::CoxMethod)
    #η = M.X*beta
    η = getX(M) * beta
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
include("Cox/CoxOptim.jl")
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
get_og_T(M::CoxMethod) = M.T[sortperm(get_perm(M))]

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
predict_risk(C::Cox; type = :risk) = exp.(predict_lp(C))

# Step-interpolate the Breslow cumulative baseline hazard `Λ0` (sampled at the sorted
# ascending grid `t_grid`) onto an arbitrary time `t`. Cumulative hazard is right-continuous
# with left limits: returns `0` for `t < t_grid[1]` and `Λ0[end]` for `t ≥ t_grid[end]`.
function _cumhaz_at(Λ0::AbstractVector, t_grid::AbstractVector, t::Real)
    idx = searchsortedlast(t_grid, t)
    return idx == 0 ? zero(eltype(Λ0)) : Λ0[idx]
end

# Internal: returns an `n × length(times)` matrix of cumulative hazards
# `Λᵢ(t) = Λ₀(t) · exp(ηᵢ)` for every subject `i` and every requested time.
function _predict_expected_at(C::Cox, times::AbstractVector)
    Λ0 = baseline_hazard(C, centered=true)
    t_grid = sort(unique(C.M.T))
    η = exp.(predict_lp(C))
    n, m = length(η), length(times)
    out = Matrix{Float64}(undef, n, m)
    @inbounds for j in 1:m
        Λ_j = _cumhaz_at(Λ0, t_grid, times[j])
        for i in 1:n
            out[i, j] = Λ_j * η[i]
        end
    end
    return out
end

"""
    predict_expected(C::Cox)
    predict_expected(C::Cox, t::Real)
    predict_expected(C::Cox, ts::AbstractVector)

Cumulative hazard `Λᵢ(t) = Λ₀(t) · exp(ηᵢ)` per subject, where `Λ₀` is the Breslow
estimator centered to the fit's reference (continuous covariates at their mean,
categorical at their mode) and `ηᵢ = (Xᵢ - X̄) β`.

Output shape:
- no time argument → length-`n` vector with each subject evaluated at their own
  observed time `Tᵢ` (the convention used by R's `predict(coxph, type="expected")`);
- `t::Real` → length-`n` vector at the scalar time;
- `ts::AbstractVector` → `n × length(ts)` matrix.
"""
function predict_expected(C::Cox)
    Λ0 = baseline_hazard(C, centered=true)
    t_grid = sort(unique(C.M.T))
    η = exp.(predict_lp(C))             # original data order
    T_og = get_og_T(C.M)                # original data order
    return [_cumhaz_at(Λ0, t_grid, T_og[i]) * η[i] for i in eachindex(η)]
end
predict_expected(C::Cox, t::Real)           = vec(_predict_expected_at(C, [t]))
predict_expected(C::Cox, ts::AbstractVector) = _predict_expected_at(C, ts)

"""
    predict_survival(C::Cox)
    predict_survival(C::Cox, t::Real)
    predict_survival(C::Cox, ts::AbstractVector)

Per-subject survival probability `Sᵢ(t) = exp(-Λᵢ(t))` derived from
[`predict_expected`](@ref). Shapes match `predict_expected`: the no-argument form
returns a length-`n` vector with each subject at their own observed time `Tᵢ`;
the `t::Real` form is a length-`n` vector at a scalar time; the `ts::AbstractVector`
form is an `n × length(ts)` matrix.
"""
predict_survival(C::Cox)                     = exp.(-predict_expected(C))
predict_survival(C::Cox, t::Real)            = exp.(-predict_expected(C, t))
predict_survival(C::Cox, ts::AbstractVector) = exp.(-predict_expected(C, ts))

# ─────────────────────────────────────────────────────────────────────────────
# Predict on new data
# ─────────────────────────────────────────────────────────────────────────────

# Apply the fit's stored schema to `newdata` and return the design matrix `X_new`.
# Errors if the model was constructed without a stored formula (e.g. by a direct
# call to `Cox(obj, names, types)` without the keyword).
function _build_X_for_newdata(C::Cox, newdata::DataFrame)
    isnothing(C.formula) && error("This Cox model was constructed without a stored formula; predict-on-newdata is not available. Re-fit via `fit(Cox, @formula(...), df)` so the formula is captured.")
    return modelcols(C.formula.rhs, newdata)
end

# Reference subject (1 × p row) from the training data: continuous covariates at
# their training mean, categorical at their training mode. Used to centre linear
# predictors at predict-time so the convention matches the no-newdata path and R.
function _training_reference(C::Cox)
    X_train = get_og_X(C.M)
    d = nvar(C.M)
    mX = zeros(1, d)
    cat = C.pred_types .== :categorical
    for i in 1:d
        mX[1, i] = cat[i] ? mode(sort(X_train[:, i])) : mean(X_train[:, i])
    end
    return mX
end

# Linear predictor on new data, centred against the training reference by default.
function predict_lp(C::Cox, newdata::DataFrame; centered::Bool=true)
    X_new = _build_X_for_newdata(C, newdata)
    η_new = X_new * C.β
    centered || return η_new
    return η_new .- (_training_reference(C) * C.β)[1]
end

# Relative risk `exp(η_new)` on new data.
predict_risk(C::Cox, newdata::DataFrame) = exp.(predict_lp(C, newdata))

# Per-subject × per-variable centred contributions to the linear predictor on new data.
function predict_terms(C::Cox, newdata::DataFrame)
    X_new = _build_X_for_newdata(C, newdata)
    β = C.β
    d = nvar(C.M)
    rez = X_new .* β'
    mX = _training_reference(C)
    @inbounds for i in 1:d
        rez[:, i] .-= mX[1, i] * β[i]
    end
    return rez
end

# Internal: `n_new × length(times)` matrix of cumulative hazards on newdata.
function _predict_expected_at_newdata(C::Cox, newdata::DataFrame, times::AbstractVector)
    Λ0 = baseline_hazard(C, centered=true)
    t_grid = sort(unique(C.M.T))
    η_new = exp.(predict_lp(C, newdata))
    n, m = length(η_new), length(times)
    out = Matrix{Float64}(undef, n, m)
    @inbounds for j in 1:m
        Λ_j = _cumhaz_at(Λ0, t_grid, times[j])
        for i in 1:n
            out[i, j] = Λ_j * η_new[i]
        end
    end
    return out
end

"""
    predict_expected(C::Cox, newdata::DataFrame, t::Real)
    predict_expected(C::Cox, newdata::DataFrame, ts::AbstractVector)

Per-subject cumulative hazard `Λᵢ(t)` for subjects in `newdata` evaluated at user-supplied
times. Length-`n_new` vector for scalar `t`; `n_new × length(ts)` matrix for a vector.
Centred against the training reference subject so the absolute scale matches the
no-newdata path. Newdata predictions require an explicit time argument — there is no
"own time" default for arbitrary new subjects.
"""
predict_expected(C::Cox, newdata::DataFrame, t::Real)            = vec(_predict_expected_at_newdata(C, newdata, [t]))
predict_expected(C::Cox, newdata::DataFrame, ts::AbstractVector) = _predict_expected_at_newdata(C, newdata, ts)

"""
    predict_survival(C::Cox, newdata::DataFrame, t::Real)
    predict_survival(C::Cox, newdata::DataFrame, ts::AbstractVector)

Per-subject survival probability `Sᵢ(t) = exp(-Λᵢ(t))` on `newdata` at user times.
Shapes match [`predict_expected`](@ref).
"""
predict_survival(C::Cox, newdata::DataFrame, t::Real)            = exp.(-predict_expected(C, newdata, t))
predict_survival(C::Cox, newdata::DataFrame, ts::AbstractVector) = exp.(-predict_expected(C, newdata, ts))

function StatsBase.predict(C::Cox, type::Symbol=:lp)
    type==:lp && return predict_lp(C)
    type==:risk && return predict_risk(C)
    type==:expected && return predict_expected(C)
    type==:terms && return predict_terms(C)
    type==:survival && return predict_survival(C)
    error("The prediction you want was not understood. Please pass a `type` parameter among (:lp, :risk, :expected, :terms, :survival). See `?Cox` for details.")
end

function StatsBase.predict(C::Cox, type::Symbol, t::Real)
    type==:expected && return predict_expected(C, t)
    type==:survival && return predict_survival(C, t)
    error("Time-indexed `predict` only supports `:expected` and `:survival`, got `:$type`.")
end

function StatsBase.predict(C::Cox, type::Symbol, ts::AbstractVector)
    type==:expected && return predict_expected(C, ts)
    type==:survival && return predict_survival(C, ts)
    error("Time-indexed `predict` only supports `:expected` and `:survival`, got `:$type`.")
end

function StatsBase.predict(C::Cox, type::Symbol, newdata::DataFrame)
    type == :lp    && return predict_lp(C, newdata)
    type == :risk  && return predict_risk(C, newdata)
    type == :terms && return predict_terms(C, newdata)
    type in (:expected, :survival) && error("`:$type` on newdata requires a time argument: predict(C, :$type, newdata, t).")
    error("Unsupported predict type `:$type`. Supported on newdata: `:lp`, `:risk`, `:terms`, `:expected`, `:survival`.")
end

function StatsBase.predict(C::Cox, type::Symbol, newdata::DataFrame, t::Real)
    type == :expected && return predict_expected(C, newdata, t)
    type == :survival && return predict_survival(C, newdata, t)
    error("Time-indexed `predict` on newdata only supports `:expected` and `:survival`, got `:$type`.")
end

function StatsBase.predict(C::Cox, type::Symbol, newdata::DataFrame, ts::AbstractVector)
    type == :expected && return predict_expected(C, newdata, ts)
    type == :survival && return predict_survival(C, newdata, ts)
    error("Time-indexed `predict` on newdata only supports `:expected` and `:survival`, got `:$type`.")
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
    return Cox(CoxWorkerType(Y, Δ, X), Symbol.(predictor_names), catego; formula = formula_applied)
end

"""
    summary(model::Cox)

Compute a statistical summary table for a fitted Cox proportional hazards model.

This function calculates standard errors, z-scores, p-values, and confidence intervals.

# Returns
A `DataFrame` with the following columns:
- `predictor`: Name of the covariate.
- `β`: Estimated regression coefficient.
- `e_β`: Hazard ratio (exp(β)).
- `se`: Standard error of the coefficient.
- `z_scores`: Wald statistic (β / se).
- `p_values`: Two-sided p-value.
- `ci_lower_β`: Lower bound of the 95% confidence interval for β.
- `ci_upper_β`: Upper bound of the 95% confidence interval for β.

# Example
```julia
model = fit(Cox, @formula(Surv(time, status) ~ age + sex), df)

# Get the full summary dataframe
summ = summary(model)

# Extract standard errors specifically
standard_errors = summ.se
```
"""
function summary(C::Cox)
    # Standard Error, z-score, p-values and c-index: 
    se = sqrt.(diag(inv(get_hessian(C.M, C.β))))
    z_scores = C.β ./ se
    p_values = 2 .* ccdf.(Normal(), abs.(z_scores))

    q = quantile(Normal(), 0.975) 
    ci_lower_β = C.β .- se .* q
    ci_upper_β = C.β .+ se .* q
    return DataFrame(
        predictor = C.pred_names,
        β = C.β,
        e_β = exp.(C.β),
        se = se,
        z_scores = z_scores,
        p_values = p_values,
        ci_lower_β  = ci_lower_β,
        ci_upper_β = ci_upper_β
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
