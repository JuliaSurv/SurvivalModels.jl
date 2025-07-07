```@meta
CurrentModule = SurvivalModels
```

# Cox models

one ref: [cox1972regression](@cite).

## Theory

The Cox Proportional Hazards Model is a semi-parametric model used to analyze time-to-event data. It models the relationship between the survival time of an individual and a set of covariates. It is defined by the **hazard function**:

```math
h(t | \mathbf{X}) = h_0(t) \exp(\mathbf{X}^T\mathbf{\beta})
```
where:
$h_0(t)$ is the baseline hazard function, 
$\mathbf{X}$ is the covariate vector, 
$\mathbf{\beta}$ is the vector of regression coefficients.

The partial-likelihood function for the Cox model is given by:

```math
L(\mathbf{\beta}) = \prod_{j=1}^{k} \frac{\prod_{ \Delta_i=1} \exp(\mathbf{X}_i^T\mathbf{\beta})}{\left( \sum_{l \in R_j} \exp(\mathbf{X}_l^T\mathbf{\beta}) \right)}
```
Our goal is to maximize the log-partial-likelihood or, equivalently, to minimize its negative, which we define as our loss function:

```math

\text{Loss}(\mathbf{\beta}) = - \log L(\mathbf{\beta}) = - \sum_{j=1}^{k} \left( \sum \mathbf{X}_i^T\mathbf{\beta} - \log \left( \sum_{l \in R_j} \exp(\mathbf{X}_l^T\mathbf{\beta}) \right) \right)

```

The loss function is coded as follows: 

```julia
function loss(beta, M::Cox)
    η = M.X*beta
    return dot(M.Δ, log.((M.T .<= M.T') * exp.(η)) .- η)
end
```

We differentiate once and obtain the gradient:

```math

\frac{\partial}{\partial \beta_k} \text{Loss}(\mathbf{\beta}) = - \sum_{i=1}^{n} \left( X_{ik} - \frac{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jk}}{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j)} \right)

```
And the Hesian matrix: 

```math

\frac{\partial^2}{\partial \beta_k \partial \beta_l} \text{Loss}(\mathbf{\beta}) = \sum_{i=1}^{n} \left[ \frac{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jk}X_{jl}}{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j)} - \frac{\left( \sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jk} \right) \left( \sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jl} \right)}{\left( \sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) \right)^2} \right]

```
The information matrix is defined as the negative of the Hessian matrix of the log-likelihood function, evaluated at the point of the estimated coefficients. 

```math

I(\hat{\boldsymbol{\beta}}) = -H(\hat{\boldsymbol{\beta}})

```

But, in the earlier formula, $\mathbf{H}_{\text{Loss}}$​ was for Loss(β), which is -log L(β) $\mathbf{H}_{\text{Loss}} = - \mathbf{H}_{\text{log-likelihood}}$. Therefore, the observed Information Matrix is equal to $\mathbf{H}_{\text{Loss}}$ itself.

The variance (and covariance) of our estimators $\hat{\boldsymbol{\beta}}$ are obtained by inverting the observed information matrix.

$$\text{Var}(\hat{\boldsymbol{\beta}}) = I(\hat{\boldsymbol{\beta}})^{-1}$$

This final matrix contains:
- On its diagonal: the variances of each coefficient ($\text{Var}(\hat{\beta}_1)$, $\text{Var}(\hat{\beta}_2)$, ...).
- Off-diagonal: the covariances between pairs of coefficients.

The standard error for a specific coefficient ($\hat{\beta}_k$) is the square root of its variance.

$$SE(\hat{\beta}_k) = \sqrt{\text{Var}(\hat{\beta}_k)}$$

To determine if a variable has a statistically significant effect, a Wald test is performed. A z-score is calculated:
$$z = \frac{\text{Coefficient}}{\text{Erreur Type}} = \frac{\hat{\beta}}{SE(\hat{\beta})}$$

This $z$-score is then compared to a normal distribution to obtain a $p$-value. A low $p$-value (typically < 0.05) suggests that the coefficient is significantly different from zero.

The p-value for each coefficient is calculated by comparing its z-score to a standard normal distribution. This p-value indicates the probability of observing a z-score as extreme as, or more extreme than, the one calculated, assuming the null hypothesis (that the coefficient is zero) is true.


The standard error allows for the construction of a confidence interval (CI) around the coefficient, which provides a range of plausible values for the true coefficient.

The general formula for a $(1 - \alpha) \times 100\%$ confidence interval is:

$$\text{IC pour } \hat{\beta} = \hat{\beta} \pm z_{\alpha/2} \times SE(\hat{\beta})$$





Let us see for example the output on the `colon` dataset: 

```@example 2
using SurvivalModels
using RDatasets

# ovarian = dataset("survival", "ovarian")
# ovarian.FUTime = Float64.(ovarian.FUTime)
# ovarian.FUStat = Bool.(ovarian.FUStat)
# model = fit(Cox, @formula(Surv(FUTime, FUStat) ~ Age + ECOG_PS), ovarian)

colon = dataset("survival", "colon")
colon.Time = Float64.(colon.Time)
colon.Status = Bool.(colon.Status)
model_colon = fit(Cox, @formula(Surv(Time, Status) ~ Age + Rx), colon)

```

The outputed datafrae contains collumns with respectively the name of the predictor, the obtained coefficients, its standard error, the associated p-value and the test statistic z as just described. 

```@docs
Cox
```


## Different versions of the optimisation routine

To implement the Cox proportional hazards model, different versions were coded, using different methods.
The final goal is to compare these versions and choose the most efficient one: the fastest and the closest to the true values
of the coefficients. 

### V0

```@docs
CoxV0
```

### V1: Implementation using the 'Optimization.jl' Julia package 

```@docs
CoxV1
```

### V2: Implementation using the gradient and the Hessian matrix

```@docs
CoxV2
```

### V3: Improved version of V2 (much faster) because non-allocative.

```@docs
CoxV3
```

### V4: Majoration of the Hessian matrix by a universal bound.

```@docs
CoxV4
```

### V5

```@docs
CoxV5
```

## Comparison of the different methods speed

We propose to compare the different methods on simulated data, with varying number of lines and columns, to verify empirically the theoretical complexity of the different methods. 
We will then compare the results with Julia's and R's existing Cox implementation.


```@example 1
using SurvivalModels, Plots, Random, Distributions, StatsBase, LinearAlgebra, DataFrames, RCall, Survival

using SurvivalModels: getβ, CoxV0, CoxV1, CoxV2, CoxV3, CoxV4, CoxV5

struct CoxVJ
    T::Vector{Float64}
    Δ::Vector{Bool}
    X::Matrix{Float64}
    function CoxVJ(T,Δ,X)
        new(T,Bool.(Δ),X)
    end
end

function SurvivalModels.getβ(M::CoxVJ)
    return fit(Survival.CoxModel, M.X, Survival.EventTime.(M.T,M.Δ)).β
end

R"""
library(survival)
"""

struct CoxVR
    df::DataFrame
    function CoxVR(T,Δ,X)
        df = DataFrame(X,:auto)
        df.status = Δ
        df.time = T
        new(df)
    end
end

function SurvivalModels.getβ(M::CoxVR)
    df = M.df
    @rput df
    R"""
    beta  <- coxph(Surv(time,status)~., data = df, ties="breslow")$coefficients
    """
    @rget beta
    return beta
end

# Creating a dictionary for all the models:
# Label => (constructor, plotting color)

const design = Dict(
"V0"=> (CoxV0, :blue),
"V1"=> (CoxV1, :orange),
"V2"=> (CoxV2, :brown),
"V3"=> (CoxV3, :purple),
"V4"=> (CoxV4, :green),
"V5"=> (CoxV5, :yellow),
"VR"=> (CoxVR, :red),
"VJ"=> (CoxVJ, :black)
);

# Function to simulate data for different rows and column numbers (n max = 2000 et m max = 20):
function simulate_survival_data(n, m; censor_rate = 0.2, β=randn(m))
    Random.seed!(42)
    X = hcat(
        [randn(n)       for _ in 1:cld(m,3)]..., # about 1/3
        [rand(n)        for _ in 1:cld(m,3)]..., # about 1/3
        [exp.(randn(n)) for _ in 1:(m-2cld(m,3))]... # the rest. 
    )
    η = X * β
    λ₀ = 1 
    U = rand(n)
    O = -log.(U) ./ (λ₀ .* exp.(η))
    lc = quantile(O, 1 - censor_rate)
    C = rand(Exponential(lc), n)
    T = min.(O, C)
    Δ = Bool.(T .<= C)
    return (T, Δ, X)
end

# Run the models and get the running time, the β coefficients and the difference between the true β and the obtained ones: 
function run_models() 
    Ns = (500, 1000, 2000) 
    Ms = (10, 20) 
    true_betas = randn(maximum(Ms))
    df = []
    for n in Ns, m in Ms
        if (n == 2000) | (m == 20) # Only if they end up in the graphs.
            data = simulate_survival_data(n,m, β = true_betas[1:m])
            for (name, (constructor, _)) in design
                display((n,m,name))
                model = constructor(data...)
                beta = getβ(model)
                time = @elapsed getβ(model)
                push!(df, (
                    n = n, 
                    m = m, 
                    name = name, 
                    time = time,
                    beta = beta,
                    diff_to_truth = sqrt(sum((beta .- true_betas[1:m]).^2)/sum(true_betas[1:m].^2)),
                ))
            end
        end
    end
    df = DataFrame(df)
    sort!(df, :name)
    return df
end

# Plot the results, starting with the time: 
function timing_graph(df)
    group1 = groupby(filter(r -> r.m==20, df), :name)
    p1 = plot(; xlabel = "Number of observations (n)",
        ylabel = "Time (in seconds)",
        yscale= :log10,
        xscale= :log10,
        title = "For m=20 covs., varying n",
        legend = :bottomright,
        lw = 1);
    for g in group1
        plot!(p1, g.n, g.time, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
    end
    group2 = groupby(filter(r -> r.n==2000, df), :name)
        p2 = plot(; xlabel = "Number of covariates (m)",
            ylabel = "Temps (ms)",
            yscale= :log10,
            xscale= :log10,
            title = "For n=2000 obs., varying m",
            legend = :bottomright,
            lw = 1);
        for g in group2
            plot!(p2, g.m, g.time, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
        end
        p = plot(p1,p2, size=(1200,600), plot_title = "Runtime (logscale) of the various implementations")
        return p
end

df = run_models()
timing_graph(df)
```

comments on the graph. 

A zoom on our implementation vs Survival.jl vs R::survival: 

```@example 1
timing_graph(filter(r -> r.name ∈ ("V3", "VJ", "VR"), df))
```

So we are about x10 faster than the reference implmentation of R (and than the previous Julia attemps) on this example. 


```@example 1
function beta_correctness_graphs(df; ref="VJ")
    
    reflines = filter(r -> r.name == ref, df)
    rename!(reflines, :beta => :refbeta)
    select!(reflines, Not([:name, :time, :diff_to_truth]))
    otherlines = filter(r -> r.name != ref, df)
    rez = leftjoin(otherlines, reflines, on=[:n,:m])
    percent(x,y) = sqrt(sum((x .- y).^2)/sum(y .^2))
    rez.error = percent.(rez.beta, rez.refbeta)
    select!(rez, [:n,:m,:name,:error])
    rez = filter!(r -> !isnan(r.error), rez)
    
    group1 = groupby(filter(r -> r.m==20, rez), :name)
    p1 = plot(; xlabel = "Number of observations (n)",
                ylabel = "L2dist to $ref's β",
                yscale=:log10,
                xscale= :log10,
                title = "m=20, varying n",
                legend = :bottomright,
                lw = 1);
    for g in group1
        plot!(p1, g.n, g.error, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
    end

    group2 = groupby(filter(r -> r.n==2000, rez), :name)
    p2 = plot(; xlabel = "Nomber of covariates (m)",
                ylabel = "L2Dist to $ref's β",
                yscale=:log10,
                xscale= :log10,
                title = "n=2000, varying m",
                legend = :bottomright,
                lw = 1);
    for g in group2
        plot!(p2, g.m, g.error, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
    end
    p = plot(p1,p2, size=(1200,600), plot_title="β-correctness w.r.t. $ref's version.")
    return p
end

beta_correctness_graphs(df)
```

```@example 1
function beta_wrt_truth(df)
    group1 = groupby(filter(r -> r.m==20, df), :name)
    p1 = plot(; xlabel = "Number of observations (n)",
                ylabel = "L2dist to the truth",
                yscale=:log10,
                xscale= :log10,
                title = "m=20, varying n",
                legend = :bottomright,
                lw = 1);
    for g in group1
        plot!(p1, g.n, g.diff_to_truth, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
    end

    group2 = groupby(filter(r -> r.n==2000, df), :name)
    p2 = plot(; xlabel = "Nomber of covariates (m)",
                ylabel = "L2Dist to the truth",
                yscale=:log10,
                xscale= :log10,
                title = "n=2000, varying m",
                legend = :bottomright,
                lw = 1);
    for g in group2
        plot!(p2, g.m, g.diff_to_truth, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
    end
    p = plot(p1,p2, size=(1200,600), plot_title="β-correctness w.r.t. the truth.")
    return p
end   

beta_wrt_truth(df)
```



```@bibliography
Pages = ["cox.md"]
Canonical = false
```