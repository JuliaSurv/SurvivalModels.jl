```@meta
CurrentModule = SurvivalModels
```

# Log-Rank Test

The log-rank test [mantel1966evaluation](@cite) is a non-parametric test to compare the survival distributions of two or more groups. It can be stratified to account for baseline differences.

Suppose we have ``G`` groups. At each event time ``t_j``:

- ``d_{gj}``: number of events in group ``g`` at ``t_j``
- ``Y_{gj}``: number at risk in group ``g`` just before ``t_j``
- ``d_j = \sum_g d_{gj}``: total events at ``t_j``
- ``Y_j = \sum_g Y_{gj}``: total at risk at ``t_j``

The expected number of events in group ``g`` at ``t_j`` under the null hypothesis is:

```math
E_{gj} = Y_{gj} \frac{d_j}{Y_j}
```

The log-rank test statistic is:

```math
Z_g = \sum_j (d_{gj} - E_{gj})
```

For two groups, the test statistic is:

```math
Z = \frac{\left[\sum_j (d_{1j} - E_{1j})\right]^2}{\sum_j V_{1j}}
```

where

```math
V_{1j} = \frac{Y_{1j} Y_{2j} d_j (Y_j - d_j)}{Y_j^2 (Y_j - 1)}
```

Under the null hypothesis, ``Z`` is approximately chi-squared distributed with ``G-1`` degrees of freedom.

## Stratified Log-Rank Test

If there are stratas, the test statistic and variance are summed over strata.

## Usage

You can compute a log-rank test using the following code: 

```@example 1
using SurvivalModels

T = [1, 2, 3, 4, 1, 2, 3, 4]
Δ = [1, 1, 1, 1, 1, 1, 1, 1]
group = [1, 1, 2, 2, 1, 1, 2, 2]
strata = [1, 1, 1, 1, 2, 2, 2, 2]
lrt = LogRankTest(T, Δ, group, strata)
```

and/or with the formula interface: 

```@example 1
using DataFrames
df = DataFrame(time=T, status=Δ, group=group, strata=strata)
lrt2 = fit(LogRankTest, @formula(Surv(time, status) ~ Strata(strata) + group), df)
```

The produced object has the following fields: 

- `stat`: Chi-square test statistic.
- `df`: Degrees of freedom.
- `pval`: P-value of the test.


## References

```@docs
LogRankTest
```

```@bibliography
Pages = ["logranktest.md"]
Canonical = false
```