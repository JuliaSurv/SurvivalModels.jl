# Fitting, prediction & simulation

## Fitting and inference

A model is fitted from data with `fit(Model, @formula(Surv(time, status) ~ ...), df)` (or the two-formula variant for `GeneralHazard`); see [Illustrative example](@ref) for complete calls. The optimizer is seeded automatically from the data.

### Fit statistics

`GeneralHazardModel <: StatsAPI.StatisticalModel`, so a fitted model supports the standard statistical-model accessors. `aic`, `aicc`, and `bic` follow from `loglikelihood`, `dof`, and `nobs`; `stderror` follows from `vcov`. `coef` and `vcov` are reported on the inference scale `[log.(baseline parameters); active regression coefficients]`, so `MvNormal(coef(m), vcov(m))` is a coherent parameter-uncertainty distribution.

```@docs
SurvivalModels.loglikelihood(::SurvivalModels.GeneralHazardModel)
SurvivalModels.nobs(::SurvivalModels.GeneralHazardModel)
SurvivalModels.dof(::SurvivalModels.GeneralHazardModel)
SurvivalModels.coef(::SurvivalModels.GeneralHazardModel)
SurvivalModels.vcov(::SurvivalModels.GeneralHazardModel)
```

### Brier score

Inverse-probability-of-censoring-weighted Brier score (Graf et al. 1999) and its integrated form work for `GeneralHazardModel` through the same `brier_score(model, ...)` / `integrated_brier_score(model, ...)` API used for Cox. See the [Model Evaluation: Brier Score](@ref) section of the Cox documentation for the mathematical definition and signature list.

## Prediction

Once a `GeneralHazardModel` is fitted (or directly constructed), you can evaluate per-subject cumulative hazards and survival probabilities at user-supplied times. The four hazard structures share the same closed-form expression via the unified representation

```math
H(t \,|\, x) = H_0\!\left(t\, c_1(x)\right) c_2(x)
```

where ``H_0`` is the cumulative hazard of the baseline distribution and ``(c_1, c_2)`` are the method-specific time- and hazard-scale multipliers (`c1`/`c2` in the code). The survival is ``S(t \,|\, x) = \exp(-H(t \,|\, x))``.

```julia
predict(model, :survival)              # length-n vector, each subject at own Tᵢ
predict(model, :expected)              # length-n vector of Λᵢ(Tᵢ)
predict(model, :survival, t)           # length-n vector at scalar t
predict(model, :expected, t)
predict(model, :survival, ts)          # n × length(ts) matrix
predict(model, :expected, ts)
```

The default no-arg form (`predict(model)` or `predict(model, :survival)`) evaluates each subject at their own observed time ``T_i``, matching the convention used by the Cox interface.

### Predict on new data

Each prediction also accepts a `newdata::DataFrame` argument. The fit's stored formula(s) are re-applied to `newdata` to rebuild the design matrices ``X_1``, ``X_2``, so `newdata` must contain every predictor column referenced in the original `@formula(...)`. For models fit with `fit(GHM, formula, df)` (one formula), the same formula is stored twice and used for both ``X_1`` and ``X_2``; for `fit(GeneralHazard, formula1, formula2, df)` the two are stored separately.

```julia
predict(model, :survival, newdata, t)        # length-n_new at scalar t
predict(model, :expected, newdata, t)
predict(model, :survival, newdata, ts)       # n_new × length(ts) matrix
predict(model, :expected, newdata, ts)
```

Newdata predict **requires** an explicit time argument — there is no "own time" default for arbitrary new subjects. Models built directly via the positional constructor (without the `formula1` / `formula2` keyword arguments) do not have stored formulas and will error on newdata predict.

```@docs
SurvivalModels.predict_expected(::SurvivalModels.GeneralHazardModel)
SurvivalModels.predict_survival(::SurvivalModels.GeneralHazardModel)
```

## Simulation

The `simulate` command (ported from `HazReg.jl`; formerly `simGH`, now a deprecated alias) allows one to simulate times to event from the following models:

- General Hazard (GH) model [chen:2001](@cite) [rubio:2019](@cite).
- Accelerated Failure Time (AFT) model [kalbfleisch:2011](@cite).
- Proportional Hazards (PH) model [cox:1972](@cite).
- Accelerated Hazards (AH) model [chen:2000](@cite).

```@docs
simulate
```
