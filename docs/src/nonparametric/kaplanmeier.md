```@meta
CurrentModule = SurvivalModels
```

# Kaplan-Meier Estimator

The Kaplan-Meier estimator[kaplan1958nonparametric](@cite) is a non-parametric statistic used to estimate the survival function from lifetime data, especially when data are censored. The Greenwood formula [greenwood1926report](@cite) is used for variance estimation.
 
Suppose we observe ``n`` individuals, with observed times ``T_1, T_2, \ldots, T_n`` and event indicators ``\Delta_1, \Delta_2, \ldots, \Delta_n`` (``\Delta_i = 1`` if the event occurred, ``0`` if censored).

Let ``t_1 < t_2 < \cdots < t_k`` be the ordered unique event times, and set: 

- ``d_j``: number of events at time ``t_j``
- ``n_j``: number of individuals at risk just before ``t_j``

The **Kaplan-Meier estimator** of the survival function ``S(t)`` is:

```math
\hat{S}(t) = \prod_{t_j \leq t} \left(1 - \frac{d_j}{n_j}\right)
```

This product runs over all event times ``t_j`` less than or equal to ``t``.

## Greenwood's Formula

The **Greenwood estimator** [greenwood1926report](@cite) for the variance of ``\hat{S}(t)`` is:

```math
\widehat{\mathrm{Var}}[\hat{S}(t)] = \hat{S}(t)^2 \sum_{t_j \leq t} \frac{d_j}{n_j (n_j - d_j)}
```

This allows for the construction of confidence intervals for the survival curve.

## How to use it

You can compute these estimators using the following code: 

```@example 1
using SurvivalModels

T = [2, 3, 4, 5, 8]
Δ = [1, 1, 0, 1, 0]
km = KaplanMeier(T, Δ)
```

and/or with the formula interface: 

```@example 1
using DataFrames
df = DataFrame(time=Float64.(T), status=Bool.(Δ))
km = fit(KaplanMeier, @formula(Surv(time, status) ~ 1), df)
```

The obtained objects has the following fields: 

- `t`: Sorted unique event times.
- `∂N`: Number of uncensored deaths at each time point.
- `Y`: Number of individuals at risk at each time point.
- `∂Λ`: Increments of cumulative hazard.
- `∂σ`: Greenwood variance increments.

The obtained object can be used to compute survival and variance estimates as follows: 

```@example 1
using SurvivalModels: greenwood
Ŝ = km(5.0)  # Survival probability at time 5
v̂ = greenwood(km, 5.0)  # Greenwood variance at time 5
Ŝ, v̂
```

Finally, a ``(1-\alpha) \times 100\%`` confidence interval for ``S(t)`` can be constructed using the log-minus-log transformation:

```math
\log(-\log \hat{S}(t)) \pm z_{1-\alpha/2} \frac{1}{\log \hat{S}(t)} \sqrt{\widehat{\mathrm{Var}}[\hat{S}(t)]}
```

The `confint` function can do it for you: 

```@example 2
using SurvivalModels

T = [2, 3, 4, 5, 8]
Δ = [1, 1, 0, 1, 0]
km = KaplanMeier(T, Δ)

# Compute confidence intervals at each event time (default 95%)
ci = confint(km)
first(ci, 5)  # show the first 5 rows
```

## References

```@docs
KaplanMeier
greenwood
```

```@bibliography
Pages = ["kaplanmeier.md"]
Canonical = false
```