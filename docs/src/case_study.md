```@meta
CurrentModule = SurvivalModels
```

# Case Study: Survival Analysis on the Colon Dataset

## Dataset Description

The `colon` dataset from the R `survival` package contains data from a clinical trial of colon cancer patients.  
Key variables include:

- `Time`: Survival or censoring time (days)
- `Status`: Event indicator (1 = death, 0 = censored)
- `Rx`: Treatment group (Obs, Lev, Lev+5FU)
- `Sex`: Sex (1 = male, 0 = female)
- `Age`: Age at entry
- `Node`: Number of positive lymph nodes
- `Extent`: Extent of local spread (1 = confined, 2 = adjacent, 3 = adherent, 4 = invaded)
- `Differ`: Tumor differentiation (1 = well, 2 = moderate, 3 = poor)
- `Perfor`: Perforation (0 = no, 1 = yes)

Let’s load and inspect the data:

```@example 1
using SurvivalModels, RDatasets, DataFrames, Plots
colon = dataset("survival", "colon")
colon.Time = Float64.(colon.Time)
colon.Status = Bool.(colon.Status)
first(colon, 5)
describe(colon)
```

## Exploratory Analysis

Let’s look at the distribution of survival times and events:

```@example 1
histogram(colon.Time, bins=50, xlabel="Time (days)", ylabel="Frequency", title="Distribution of Survival Times")
sum(colon.Status), length(colon.Status) - sum(colon.Status) # events, censored
```

Let’s check the treatment groups and other covariates:

```@example 1
unique(colon.Rx)
combine(groupby(colon, :Rx), nrow)
```

## Kaplan-Meier Estimator

Estimate and plot the overall survival curve:

```@example 1
km = fit(KaplanMeier, @formula(Surv(Time, Status) ~ 1), colon)
plot(km.t, cumprod(1 .- km.∂Λ), title = "Kaplan-Meier estimator", xlabel="Time (days)", ylabel="Survival probability")
```

Compare survival curves by treatment group:

```@example 1
km_obs = fit(KaplanMeier, @formula(Surv(Time, Status) ~ 1), colon[colon.Rx .== "Obs", :])
km_lev = fit(KaplanMeier, @formula(Surv(Time, Status) ~ 1), colon[colon.Rx .== "Lev", :])
km_lev5fu = fit(KaplanMeier, @formula(Surv(Time, Status) ~ 1), colon[colon.Rx .== "Lev+5FU", :])
plot(km_obs.t, cumprod(1 .- km_obs.∂Λ), label="Obs", xlabel="Time (days)", ylabel="Survival probability")
plot!(km_lev.t, cumprod(1 .- km_lev.∂Λ), label="Lev")
plot!(km_lev5fu.t, cumprod(1 .- km_lev5fu.∂Λ), label="Lev+5FU", title="KM by Treatment Group")
```

Regrouping the two similar curves, we get (including confidence intervals): 

Let’s regroup `"Lev"` and `"Lev+5FU"` as a single treatment group, and compare it to `"Obs"`.

```@example 1
# Create a new treatment variable
colon.TreatGroup = ifelse.(colon.Rx .== "Obs", "Obs", "Lev+5FU or Lev")

# Fit KM for each group
km_obs = fit(KaplanMeier, @formula(Surv(Time, Status) ~ 1), colon[colon.TreatGroup .== "Obs", :])
km_levgroup = fit(KaplanMeier, @formula(Surv(Time, Status) ~ 1), colon[colon.TreatGroup .== "Lev+5FU or Lev", :])

# Get survival and confidence intervals from the package
ci_obs = confint(km_obs)
ci_lev = confint(km_levgroup)

# Plot
plot(ci_obs.time, ci_obs.surv, ribbon=(ci_obs.surv .- ci_obs.lower, ci_obs.upper .- ci_obs.surv),
     label="Obs", color=:blue, xlabel="Time (days)", ylabel="Survival probability",
     title="KM by Regrouped Treatment", legend=:bottomleft)
plot!(ci_lev.time, ci_lev.surv, ribbon=(ci_lev.surv .- ci_lev.lower, ci_lev.upper .- ci_lev.surv),
      label="Lev+5FU or Lev", color=:red)
```

This plot shows the Kaplan-Meier survival curves for the two treatment groups, with 95% confidence intervals.

## Log-Rank Test

Test for differences in survival between treatment groups:

```@example 1
lrt = fit(LogRankTest, @formula(Surv(Time, Status) ~ TreatGroup), colon)
lrt.stat, lrt.pval
```

Interpretation: A small p-value suggests significant differences in survival between groups.

## Cox Proportional Hazards Model

Fit a Cox model with treatment and other covariates:

```@example 1
fit(Cox, @formula(Surv(Time, Status) ~ TreatGroup + Sex + Age), colon)
```

Interpretation: The coefficients show the log hazard ratios for each covariate.

## General Hazard Models (GHMs)

TODO

## Model Comparison

Let’s compare the quality of the fits using log-likelihood and AIC:
 
TODO

## Performance (Optional)

You can also compare computation times:

TODO

## Summary

- The colon dataset provides a rich set of covariates for survival analysis.
- Kaplan-Meier curves show overall and group-wise survival.
- Log-rank tests confirm differences between treatment groups.
- Cox and General Hazard Models allow for multivariate modeling and flexible hazard shapes.

*This case study can be extended with further diagnostics, stratified analyses, and more advanced parametric models as needed.*