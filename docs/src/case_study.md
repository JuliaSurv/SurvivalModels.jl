```@meta
CurrentModule = SurvivalModels
```

# Case study: the colon dataset. 

Consider the following dataset: 

```@example 1
using SurvivalModels, RDatasets, Plots
colon = dataset("survival", "colon")
colon.Time = Float64.(colon.Time)
colon.Status = Bool.(colon.Status)
names(colon)
```

We can first get the Kaplan-Meier estimates. Note that the object only stores `∂Λ`, the estimated hazards, so that you have to use the folliwng to get survival rates: 

```@example 1
km = fit(KaplanMeier, @formula(Surv(Time, Status)~1), colon)
plot(km.t, cumprod(1 .- km.∂Λ), title = "Kaplan-Meier estimator", xlabel="Time t", ylabel="Survival curve S(t)")
```

Alternatively, you can also consider the whole object `km` as being the survival function: 

```@example 1
[km(1000), km(2000)]
```

More generally: 
```@example 1
all(km.(km.t[2:end]) .== cumprod(1 .- km.∂Λ)[1:end-1])
```

Back to our example. What does this thing show ? 

What variable can we use ? 


*This case study is still a work in progress and should be completed with log-rank tests, cox, and other type of analysis...* 