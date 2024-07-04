abstract type AbstractGHMethod end
struct PHMethod  <: AbstractGHMethod end
struct AFTMethod <: AbstractGHMethod end
struct AHMethod  <: AbstractGHMethod end
struct GHMethod  <: AbstractGHMethod end

c1(::GHMethod,  X1, X2, β, α) = exp.(X2 * α)
c1(::PHMethod,  X1, X2, β, α) = 1.0
c1(::AFTMethod, X1, X2, β, α) = exp.(X1 * β)
c1(::AHMethod,  X1, X2, β, α) = exp.(X2 * α)

c2(::GHMethod,  X1, X2, β, α) = exp.(X1 * β - X2 * α)
c2(::PHMethod,  X1, X2, β, α) = exp.(-X1 * β)
c2(::AFTMethod, X1, X2, β, α) = 1.0
c2(::AHMethod,  X1, X2, β, α) = exp.(-X2 * α)

"""
    ProportionalHazard(     T, Δ, baseline, X1, X2, optimizer)
    AcceleratedFaillureTime(T, Δ, baseline, X1, X2, optimizer)
    AcceleratedHazard(      T, Δ, baseline, X1, X2, optimizer)
    GeneralHazard(          T, Δ, baseline, X1, X2, optimizer)

Maximum likelihood estimation in General Hazards models using provided `baseline` distribution, provided hazard structure (through the `method` argument), provided design matrices and given optimizer form Optim.jl. 

Parameters `T,Δ` represent observed times and statuses, while `X1, X2` should contain covariates. The number of columns in design matrices can be zero. 

Hazard structures are defined by the method, which should be `<:AbstractGHMethod`, availiable possbilities are `PHMethod()`, `AFTMethod()`, `AHMethod()` and `GHMethod()`.

The baseline distribution should be provided as a `<:Distributions.ContinuousUnivariateDitribution` object from `Distributions.jl` or compliant, e.g. from `SurvivalDistributions.jl`

method: one of NelderMead(), Newton(), LBFGS(), ConjugateGradient() or GradientDescent() or any other method taken by Optim.optimize().

maxit: maximum number of iterations of the optimization routine. 

References: 
* [Link to my reference so that people understand what it is](https://myref.com)
"""
struct GeneralHazardModel{Method, B}
    T::Vector{Float64}
    Δ::Vector{Bool}
    baseline::B
    X1::Matrix{Float64}
    X2::Matrix{Float64}
    α::Vector{Float64}
    β::Vector{Float64}
    function GeneralHazardModel(m::Method, T, Δ, baseline, X1, X2, optimizer) where Method<:AbstractGHMethod
        npd, p, q = length(Distributions.params(baseline())), size(X1,2), size(X2,2)
        init = zeros(npd+p+q)
        function mloglik(par::Vector)
            d, α, β = baseline(exp.(par[1:npd])...), par[npd .+ (1:q)], par[npd + q .+ (1:p)]
            B = (Method == AHMethod) ? 0.0 : (X1[Δ,:] * β)
            C = c1(m, X1, X2, β, α)
            D = c2(m, X1, X2, β, α)
            return  -sum(loghaz.(d, T[Δ] .* C[Δ]) .+ B) + sum(cumhaz.(d, T .* C) .* D)
        end
        par = optimize(mloglik, init, method=optimizer).minimizer
        d, α, β = baseline(exp.(par[1:npd])...), par[npd .+ (1:q)], par[npd + q .+ (1:p)]
        return new{Method, typeof(d)}(T, Δ, d, X1, X2, α, β)
    end
end

const ProportionalHazard{B}      = GeneralHazardModel{PHMethod,  B}
const AcceleratedFaillureTime{B} = GeneralHazardModel{AFTMethod, B}
const AcceleratedHazard{B}       = GeneralHazardModel{AHMethod,  B}
const GeneralHazard{B}           = GeneralHazardModel{GHMethod,  B}

ProportionalHazard(     args...; kwargs...) = GeneralHazardModel(PHMethod(),  args...; kwargs...)
AcceleratedFaillureTime(args...; kwargs...) = GeneralHazardModel(AFTMethod(), args...; kwargs...)
AcceleratedHazard(      args...; kwargs...) = GeneralHazardModel(AHMethod(),  args...; kwargs...)
GeneralHazard(          args...; kwargs...) = GeneralHazardModel(GHMethod(),  args...; kwargs...)

"""
    simGH(n, model::GeneralHazardModel)

This function simulate times to event from a general hazard model, whatever the structure it has (AH, AFT, PH, GH), and whatever its baseline distribution. 

Returns a vector containing the simulated times to event

References: 
* [HazReg original code](https://github.com/FJRubio67/HazReg) 
"""
function simGH(n, m::GeneralHazardModel{M,B}) where {M,B}
    args = (M(), m.X1, m.X2, m.β, m.α)
    p0 = 1 .- exp.(log.(1 .- rand(n)) ./ c2(args...))
    return quantile.(dist,p0) ./ c1(args...)
end