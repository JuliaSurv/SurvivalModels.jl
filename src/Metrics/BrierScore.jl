# Inverse-probability-of-censoring-weighted (IPCW) Brier score for survival models,
# per Graf, Schmoor, Sauerbrei & Schumacher (1999).

# Internal: IPCW Brier contribution summed over a single time `t`, given a pre-fit
# censoring KM. Subjects censored before `t` contribute 0.
function _brier_score_at(times, statuses, predicted_survival, t::Real, cens_km::KaplanMeier)
    Ĝ_t = cens_km(t)
    bs = 0.0
    n = length(times)
    @inbounds for i in 1:n
        Tᵢ, δᵢ, Ŝᵢ = times[i], statuses[i], predicted_survival[i]
        if Tᵢ ≤ t && δᵢ
            Ĝ_Ti = cens_km(Tᵢ)
            bs += Ĝ_Ti > 0 ? (0 - Ŝᵢ)^2 / Ĝ_Ti : 0.0
        elseif Tᵢ > t
            bs += Ĝ_t > 0 ? (1 - Ŝᵢ)^2 / Ĝ_t : 0.0
        end
        # else: Tᵢ ≤ t, δᵢ = false (censored before t) → contributes 0
    end
    return bs / n
end

"""
    brier_score(times, statuses, predicted_survival, t)
    brier_score(C::Cox, t)
    brier_score(C::Cox, ts::AbstractVector)
    brier_score(C::Cox, newdata::DataFrame, t)
    brier_score(C::Cox, newdata::DataFrame, ts::AbstractVector)

Inverse-probability-of-censoring-weighted (IPCW) Brier score per Graf et al. 1999.

# Low-level signature

`brier_score(times, statuses, predicted_survival, t)` returns the scalar Brier score at
time `t` for predicted survival probabilities `predicted_survival[i] ≈ Ŝ(t | xᵢ)`. The
censoring distribution `Ĝ` is estimated by a [`KaplanMeier`](@ref) fit on `(times,
.!statuses)` — censoring events become "observations" so the estimator gives `Ĝ(t) =
P(not censored by time t)`.

The IPCW formula:

```
BS(t*) = (1/n) Σᵢ wᵢ
```

with per-subject weights

- `(0 - Ŝᵢ)² / Ĝ(Tᵢ)`     if `Tᵢ ≤ t*` and `δᵢ = 1` (event before t*)
- `(1 - Ŝᵢ)² / Ĝ(t*)`     if `Tᵢ > t*` (still at risk at t*)
- `0`                       if `Tᵢ ≤ t*` and `δᵢ = 0` (censored before t*)

# Cox conveniences

The `C::Cox` forms compute the predicted survival via `predict(C, :survival, t)` on
training data (no `newdata` argument) or `predict(C, :survival, newdata, t)` on new
data. Vector-`ts` forms return a `Vector{Float64}` of Brier scores, one per requested
time. The censoring KM is fit once per call.
"""
function brier_score(times, statuses, predicted_survival, t::Real)
    n = length(times)
    @assert length(statuses) == n
    @assert length(predicted_survival) == n
    cens_km = KaplanMeier(times, .!statuses)
    return _brier_score_at(times, statuses, predicted_survival, t, cens_km)
end

function brier_score(C::Cox, t::Real)
    T = get_og_T(C.M)
    Δ = get_og_Δ(C.M)
    return brier_score(T, Δ, predict_survival(C, t), t)
end

function brier_score(C::Cox, ts::AbstractVector)
    T = get_og_T(C.M)
    Δ = get_og_Δ(C.M)
    cens_km = KaplanMeier(T, .!Δ)
    Ŝ = predict_survival(C, ts)   # n × length(ts)
    return [_brier_score_at(T, Δ, view(Ŝ, :, k), ts[k], cens_km) for k in eachindex(ts)]
end

function brier_score(C::Cox, newdata::DataFrame, t::Real)
    isnothing(C.formula) && error("`brier_score` on newdata requires a stored formula. Re-fit via `fit(Cox, @formula(...), df)`.")
    resp = modelcols(C.formula.lhs, newdata)
    T = Float64.(resp[:, 1])
    Δ = Bool.(resp[:, 2])
    return brier_score(T, Δ, predict_survival(C, newdata, t), t)
end

function brier_score(C::Cox, newdata::DataFrame, ts::AbstractVector)
    isnothing(C.formula) && error("`brier_score` on newdata requires a stored formula. Re-fit via `fit(Cox, @formula(...), df)`.")
    resp = modelcols(C.formula.lhs, newdata)
    T = Float64.(resp[:, 1])
    Δ = Bool.(resp[:, 2])
    cens_km = KaplanMeier(T, .!Δ)
    Ŝ = predict_survival(C, newdata, ts)
    return [_brier_score_at(T, Δ, view(Ŝ, :, k), ts[k], cens_km) for k in eachindex(ts)]
end

# Trapezoid rule ∫_x[1]^x[end] y(x) dx
function _trapezoid(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) ≥ 2
    s = 0.0
    @inbounds for i in 1:length(x)-1
        s += 0.5 * (x[i+1] - x[i]) * (y[i] + y[i+1])
    end
    return s
end

"""
    integrated_brier_score(C::Cox; t_max, n_grid=100)
    integrated_brier_score(C::Cox, newdata::DataFrame; t_max, n_grid=100)

Trapezoid-integrated Brier score over `[0, t_max]` divided by `t_max`. Computes
[`brier_score`](@ref) on a uniform grid of `n_grid` time points and applies the
trapezoid rule. The `/ t_max` normalisation makes the result comparable across
different horizon choices.
"""
function integrated_brier_score(C::Cox; t_max::Real, n_grid::Integer=100)
    grid = collect(range(0.0, Float64(t_max); length=n_grid))
    return _trapezoid(grid, brier_score(C, grid)) / t_max
end

function integrated_brier_score(C::Cox, newdata::DataFrame; t_max::Real, n_grid::Integer=100)
    grid = collect(range(0.0, Float64(t_max); length=n_grid))
    return _trapezoid(grid, brier_score(C, newdata, grid)) / t_max
end
