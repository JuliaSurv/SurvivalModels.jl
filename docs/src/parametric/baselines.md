# Baseline hazards

The hazard and the cumulative hazard functions play a crucial role in survival analysis. These functions define the likelihood function in the presence of censored observations. Thus, they are important in many contexts. For more information about these functions, see [Short course on Parametric Survival Analysis](https://github.com/FJRubio67/ShortCourseParamSurvival).

In Julia, hazard and cumulative hazard functions can be fetched through the `hazard(dist, t)` and `cumhazard(dist, t)` functions from `SurvivalDistributions.jl`, and can be applied to any distribution compliant with `Distributions.jl`'s API. Note that `SurvivalDistributions.jl` also contains a few more distributions relevant to survival analysis. See also the (deprecated) [HazReg.jl Julia Package](https://github.com/FJRubio67/HazReg.jl).

The baseline hazards commonly used with the [model classes](@ref "Model classes") (and supported by the [`simulate`](@ref) function) include:

- [Power Generalised Weibull](http://rpubs.com/FJRubio/PGW) (PGW) distribution.
- [Exponentiated Weibull](http://rpubs.com/FJRubio/EWD) (EW) distribution.
- [Generalised Gamma](http://rpubs.com/FJRubio/GG) (GenGamma) distribution.
- [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) (Gamma) distribution.
- [Lognormal](https://en.wikipedia.org/wiki/Log-normal_distribution) (LogNormal) distribution.
- [Log-logistic](https://en.wikipedia.org/wiki/Log-logistic_distribution) (LogLogistic) distribution.
- [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) (Weibull) distribution (only for AFT, PH, and AH models — see the General Hazard identifiability note).

Here are a few plots of hazard and cumulative-hazard curves for some of these distributions:

```@example 1
using Distributions, Plots, StatsBase, SurvivalDistributions
function hazard_cumhazard_plot(dist, distname; tlims=(0,10))
      plt1 = plot(t -> hazard(dist, t),
            xlabel = "x", ylabel = "Hazard", title = "$distname distribution",
            xlims = tlims, xticks = tlims[1]:1:tlims[2], label = "",
            xtickfont = font(16, "Courier"), ytickfont = font(16, "Courier"),
            xguidefontsize=18, yguidefontsize=18, linewidth=3,
            linecolor = "blue")
      plt2 = plot(t -> cumhazard(dist, t),
            xlabel = "x", ylabel = "Cumulative Hazard", title = "$distname distribution",
            xlims = tlims, xticks = tlims[1]:1:tlims[2], label = "",
            xtickfont = font(16, "Courier"), ytickfont = font(16, "Courier"),
            xguidefontsize=18, yguidefontsize=18, linewidth=3,
            linecolor = "blue")
      return plot(plt1, plt2)
end
```

## LogNormal

```@example 1
hazard_cumhazard_plot(LogNormal(0.5, 1), "LogNormal")
```

## LogLogistic

```@example 1
hazard_cumhazard_plot(Distributions.LogLogistic(1, 0.5), "LogLogistic")
```

## Weibull

```@example 1
hazard_cumhazard_plot(Weibull(3, 0.5), "Weibull")
```

## Gamma

```@example 1
hazard_cumhazard_plot(Gamma(3, 0.5), "Gamma")
```
