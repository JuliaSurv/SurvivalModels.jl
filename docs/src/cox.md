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



```math

\frac{\partial}{\partial \beta_k} \text{Loss}(\mathbf{\beta}) = - \sum_{i=1}^{n} \left( X_{ik} - \frac{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jk}}{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j)} \right)

```

```math

\frac{\partial^2}{\partial \beta_k \partial \beta_l} \text{Loss}(\mathbf{\beta}) = \sum_{i=1}^{n} \left[ \frac{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jk}X_{jl}}{\sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j)} - \frac{\left( \sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jk} \right) \left( \sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) X_{jl} \right)}{\left( \sum_{j \in R_i} \exp(\mathbf{\beta}^T\mathbf{X}_j) \right)^2} \right]

```



## Different versions of the optimisation routine

To implement the Cox proportional hazards model, different versions were coded, using different methods.
The final goal is to compare these versions and choose the most efficient one: the fastest and the closest to the true values
of the coefficients. 

### V1: Implementation using the 'Optimization.jl' Julia package 

### V2: Implementation using the gradient and the Hessian matrix

### V3: Improved version of V2 (much faster) because non-allocative.

### V4: Majoration of the Hessian matrix by a universal bound.


## Comparison of the different methods speed

We propose to compare the different methods on simulated data, with varying number of lines and columns, to verify empirically the theoretical complexity of the different methods. 

```@example 1
x = 10
# you can import packages and do the comparison here. Please comment the code outside of code boxes to ensure that the file remains readable by a human :) 
```



```@bibliography
Pages = ["cox.md"]
Canonical = false
```