# Some hazard functions in Julia

# Hazard and cumulative hazard functions

The hazard and the cumulative hazard functions play a crucial role in survival analysis. These functions define the likelihood function in the presence of censored observations. Thus, they are important in many context. For more information about these functions, see [Short course on Parametric Survival Analysis
](https://github.com/FJRubio67/ShortCourseParamSurvival).

This tutorial contains the implementation of the hazard and cumulative hazard functions for some popular distributions using the Julia package `HazReg.jl`, such as the LogNormal, LogLogistic, Weibull, and Gamma distributions. 

The Julia implementation of other, less common, distributions that are of interest in survival analysis can be found in the Julia package `HazReg.jl`: [`HazReg.jl`](https://github.com/FJRubio67/HazReg.jl).

See also: 

- [HazReg.jl Julia Package](https://github.com/FJRubio67/HazReg.jl) 

# Required packages
```@example 1
using Distributions
using Plots
using StatsBase
using HazReg
```

# Examples

## LogNormal

```@example 1
#= Hazard function =#
plot(t -> hazard(LogNormal(0.5, 1), t),
      xlabel = "x", ylabel = "Hazard", title = "LogNormal distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```

```@example 1
#= Cumulative Hazard function =#
plot(t -> cumhazard(LogNormal(0.5, 1),t),
      xlabel = "x", ylabel = "Cumulative Hazard", title = "LogNormal distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```


## LogLogistic

```@example 1
#= Hazard function =#
plot(t -> hazard(LogLogistic(1, 0.5),t),
      xlabel = "x", ylabel = "Hazard", title = "LogLogistic distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```

```@example 1
#= Cumulative Hazard function =#
plot(t -> cumhazard(LogLogistic(1, 0.5),t),
      xlabel = "x", ylabel = "Cumulative Hazard", title = "LogLogistic distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```

## Weibull

```@example 1
#= Weibull function =#
plot(t -> hazard(Weibull(3, 0.5),t),
      xlabel = "x", ylabel = "Hazard", title = "Weibull distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```

```@example 1
#= Cumulative Hazard function =#
plot(t -> cumhazard(Weibull(3, 0.5),t),
      xlabel = "x", ylabel = "Cumulative Hazard", title = "Weibull distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```


## Gamma

```@example 1
#= Weibull function =#
plot(t -> hazard(Gamma(3, 0.5),t),
      xlabel = "x", ylabel = "Hazard", title = "Gamma distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```

```@example 1
#= Cumulative Hazard function =#
plot(t -> cumhazard(Gamma(3, 0.5),t),
      xlabel = "x", ylabel = "Cumulative Hazard", title = "Gamma distribution",
    xlims = (0,10),   xticks = 0:1:10, label = "", 
    xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
    xguidefontsize=18, yguidefontsize=18, linewidth=3,
    linecolor = "blue")
```



# Julia simGH: simulating times to event from a general hazard structure

# The `simGH` command.
The simGH command from the `HazReg.jl` Julia package allows one to simulate times to event from the following models:

- General Hazard (GH) model [chen:2001](@cite) [rubio:2019](@cite).

- Accelerated Failure Time (AFT) model [kalbfleisch:2011](@cite).

- Proportional Hazards (PH) model [cox:1972](@cite).

- Accelerated Hazards (AH) model [chen:2000](@cite).


A description of these hazard models is presented below as well as the available baseline hazards.

## General Hazard model
The GH model is formulated in terms of the hazard structure

$$h(t; \alpha, \beta, \theta, {\bf x}) = h_0\left(t  \exp\{\tilde{\bf x}^{\top}\alpha\}; \theta\right) \exp\{{\bf x}^{\top}\beta\}.$$

where ${\bf x}\in{\mathbb R}^p$ are the covariates that affect the hazard level; $\tilde{\bf x} \in {\mathbb R}^q$ are the covariates the affect the time level (typically $\tilde{\bf x} \subset {\bf x}$); $\alpha \in {\mathbb R}^q$ and $\beta \in {\mathbb R}^p$ are the regression coefficients; and $\theta \in \Theta$ is the vector of parameters of the baseline hazard $h_0(\cdot)$.

This hazard structure leads to an identifiable model as long as the baseline hazard is not a hazard associated to a member of the Weibull family of distributions [chen:2001](@cite). 

## Accelerated Failure Time (AFT) model
The AFT model is formulated in terms of the hazard structure

$$h(t; \beta, \theta, {\bf x}) = h_0\left(t  \exp\{{\bf x}^{\top}\beta\}; \theta\right) \exp\{{\bf x}^{\top}\beta\}.$$

where ${\bf x}\in{\mathbb R}^p$ are the available covariates; $\beta \in {\mathbb R}^p$ are the regression coefficients; and $\theta \in \Theta$ is the vector of parameters of the baseline hazard $h_0(\cdot)$.

## Proportional Hazards (PH) model
The PH model is formulated in terms of the hazard structure

$$h(t; \beta, \theta, {\bf x}) = h_0\left(t ; \theta\right) \exp\{{\bf x}^{\top}\beta\}.$$

where ${\bf x}\in{\mathbb R}^p$ are the available covariates; $\beta \in {\mathbb R}^p$ are the regression coefficients; and $\theta \in \Theta$ is the vector of parameters of the baseline hazard $h_0(\cdot)$.

## Accelerated Hazards (AH) model
The AH model is formulated in terms of the hazard structure

$$h(t; \alpha, \theta, \tilde{\bf x}) = h_0\left(t \exp\{\tilde{\bf x}^{\top}\alpha\}; \theta\right) .$$

where $\tilde{\bf x}\in{\mathbb R}^q$ are the available covariates; $\alpha \in {\mathbb R}^q$ are the regression coefficients; and $\theta \in \Theta$ is the vector of parameters of the baseline hazard $h_0(\cdot)$.

# Available baseline hazards
The current version of the `simGH` command implements the following parametric baseline hazards for the models discussed in the previous section.

- [Power Generalised Weibull](http://rpubs.com/FJRubio/PGW) (PGW) distribution.

- [Exponentiated Weibull](http://rpubs.com/FJRubio/EWD) (EW) distribution.

- [Generalised Gamma](http://rpubs.com/FJRubio/GG) (GenGamma) distribuiton.

- [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) (Gamma) distribution.

- [Lognormal](https://en.wikipedia.org/wiki/Log-normal_distribution) (LogNormal) distribution.

- [Log-logistic](https://en.wikipedia.org/wiki/Log-logistic_distribution) (LogLogistic) distribution.

- [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) (Weibull) distribution. (only for AFT, PH, and AH models)


# Illustrative example: Julia code
In this example, we simulate $n=1,000$ times to event from the GH, PH, AFT, and AH models with PGW baseline hazards, using the Julia `simGH` command from the `HazReg` package. We censor these samples at a fixed value, and fit the corresponding models using the Julia package `HazReg`.

See also: 

- [HazReg.jl Julia Package](https://github.com/FJRubio67/HazReg.jl) 

## PGW-GH model

```@example 1
# Required packages
using HazReg
using Distributions
using Random
using DataFrames
using Optim

# Sample size
n = 10000

# Simulated design matrices
Random.seed!(123)
dist = Normal()
des = hcat(rand(dist, n), rand(dist, n))
des_t = rand(dist, n)


#----------------------------
# PGW-GH simulation
#----------------------------

# True parameters
theta0 = [0.1,2.0,5.0]
alpha0 = 0.5
beta0 = [-0.5,0.75]

# censoring
cens = 10

# Data simulation
simdat = simGH(seed = 1234, n = n, des = des, des_t = des_t,
      alpha = alpha0, beta = beta0, 
      hstr = "GH", dist = PowerGeneralizedWeibull(theta0...))

# status variable
status = collect(Bool,(simdat .< cens))

# Inducing censoring
simdat = min.(simdat, cens)

# Model fit
OPTPGWGH = GHMLE(init = fill(0.0, 3 + 1 + size(des)[2]), times = simdat,
            status = status, hstr = "GH", dist = PowerGeneralizedWeibull, 
            des = des, des_t = des_t, method = NelderMead(), maxit = 1000)

MLEPGWGH = [exp(OPTPGWGH[1].minimizer[j]) for j in 1:3] 
append!(MLEPGWGH, OPTPGWGH[1].minimizer[4:end]) 

# True parameter values vs. MLE
COMP =  hcat(vcat(theta0,alpha0,beta0),MLEPGWGH);  
COMP = DataFrame(COMP, :auto);
 
rename!( COMP, ["True", "MLE"] );
println(COMP)
```



## PGW-PH model

```@example 1

# Sample size
n = 10000

# Simulated design matrices
Random.seed!(123)
dist = Normal()
des = hcat(rand(dist, n), rand(dist, n))


#----------------------------
# PGW-PH simulation
#----------------------------

# True parameters
theta0 = [0.1,2.0,5.0]
beta0 = [-0.5,0.75]

# censoring
cens = 10

# Data simulation
simdat = simGH(seed = 1234, n = n, des = des, des_t = nothing,
      alpha = nothing, beta = beta0, 
      hstr = "PH", dist = PowerGeneralizedWeibull(theta0...))

# status variable
status = collect(Bool,(simdat .< cens))

# Inducing censoring
simdat = min.(simdat, cens)

# Model fit
OPTPGWPH = GHMLE(init = fill(0.0, 3 + size(des)[2]), times = simdat,
            status = status, hstr = "PH", dist = PowerGeneralizedWeibull, 
            des = des, des_t = nothing, method = NelderMead(), maxit = 1000)

MLEPGWPH = [exp(OPTPGWPH[1].minimizer[j]) for j in 1:3] 
append!(MLEPGWPH, OPTPGWPH[1].minimizer[4:end]) 

# True parameter values vs. MLE
COMP =  hcat(vcat(theta0,beta0),MLEPGWPH);  
COMP = DataFrame(COMP, :auto);
 
rename!( COMP, ["True", "MLE"] );
println(COMP)
```


## PGW-AFT model

```@example 1
# Sample size
n = 10000

# Simulated design matrices
Random.seed!(123)
dist = Normal()
des = hcat(rand(dist, n), rand(dist, n))


#----------------------------
# PGW-AFT simulation
#----------------------------

# True parameters
theta0 = [0.1,2.0,5.0]
beta0 = [-0.5,0.75]

# censoring
cens = 10

# Data simulation
simdat = simGH(seed = 1234, n = n, des = des, des_t = nothing,
      alpha = nothing, beta = beta0, 
      hstr = "AFT", dist = PowerGeneralizedWeibull(theta0...))

# status variable
status = collect(Bool,(simdat .< cens))

# Inducing censoring
simdat = min.(simdat, cens)

# Model fit
OPTPGWAFT = GHMLE(init = fill(0.0, 3 + size(des)[2]), times = simdat,
            status = status, hstr = "AFT", dist = PowerGeneralizedWeibull, 
            des = des, des_t = nothing, method = NelderMead(), maxit = 1000)

MLEPGWAFT = [exp(OPTPGWAFT[1].minimizer[j]) for j in 1:3] 
append!(MLEPGWAFT, OPTPGWAFT[1].minimizer[4:end]) 

# True parameter values vs. MLE
COMP =  hcat(vcat(theta0,beta0),MLEPGWAFT);  
COMP = DataFrame(COMP, :auto);
 
rename!( COMP, ["True", "MLE"] );
println(COMP)
```



## PGW-AH model

```@example 1

# Sample size
n = 10000

# Simulated design matrices
Random.seed!(123)
dist = Normal()
des_t = hcat(rand(dist, n), rand(dist, n))


#----------------------------
# PGW-AH simulation
#----------------------------

# True parameters
theta0 = [0.1,2.0,5.0]
alpha0 = [-0.5,0.75]

# censoring
cens = 10

# Data simulation
simdat = simGH(seed = 1234, n = n, des = nothing, des_t = des_t,
      alpha = alpha0, beta = nothing, 
      hstr = "AH", dist = PowerGeneralizedWeibull(theta0...))

# status variable
status = collect(Bool,(simdat .< cens))

# Inducing censoring
simdat = min.(simdat, cens)

# Model fit
OPTPGWAH = GHMLE(init = fill(0.0, 3 + size(des)[2]), times = simdat,
            status = status, hstr = "AH", dist = PowerGeneralizedWeibull, 
            des = nothing, des_t = des_t, method = NelderMead(), maxit = 1000)

MLEPGWAH = [exp(OPTPGWAH[1].minimizer[j]) for j in 1:3] 
append!(MLEPGWAH, OPTPGWAH[1].minimizer[4:end]) 

# True parameter values vs. MLE
COMP =  hcat(vcat(theta0,beta0),MLEPGWAH);  
COMP = DataFrame(COMP, :auto);
 
rename!( COMP, ["True", "MLE"] );
println(COMP)
```

```@bibliography
Pages = ["simGHJulia.md"]
Canonical = false
```