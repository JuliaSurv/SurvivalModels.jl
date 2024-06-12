module SurvivalModels

# Write your package code here.

using SurvivalBase: Surv, Strata
using StatsAPI
using StatsBase
using StatsModels

include("NonParametric/KaplanMeier.jl")
include("NonParametric/LogRankTest.jl")

export KaplanMeier, LogRankTest


function StatsBase.fit(::Type{E}, formula::FormulaTerm, df::DataFrame, rt::RateTables.AbstractRateTable) where {E<:Union{NPNSEstimator, Nessie}}
    rate_predictors = _get_rate_predictors(rt,df)
    formula_applied = apply_schema(formula,schema(df))

    if isa(formula.rhs, ConstantTerm) # No predictors
        resp = modelcols(formula_applied.lhs, df)
        return E(resp[:,1], resp[:,2], df.age, df.year, select(df,rate_predictors), rt)
    else
        # we could simply group by the left side and apply fit() again, that would make sense. 

        gdf = groupby(df, StatsModels.termnames(formula.rhs))
        return rename(combine(gdf, dfᵢ -> begin
                resp2 = modelcols(formula_applied.lhs, dfᵢ)
                E(resp2[:,1], resp2[:,2], dfᵢ.age, dfᵢ.year, select(dfᵢ, rate_predictors), rt)
            end
        ), :x1 => :estimator)
    end
end

function StatsAPI.confint(npe::E; level::Real=0.05) where E <: NPNSEstimator
    χ = sqrt(quantile(Chisq(1),1-level))
    return map(npe.Sₑ, npe.σₑ) do Sₑ,σₑ
        ci_low = exp.(log.(Sₑ) - σₑ * χ)
        ci_up = exp.(log.(Sₑ) + σₑ * χ)
        ci_low, ci_up
    end
end

function Base.show(io::IO, npe::E) where E <: NPNSEstimator
    compact = get(io, :compact, false)
    if !compact
        print(io, "$(E)(t ∈ $(extrema(npe.grid))) with summary stats:\n ")
        lower_bounds = [lower[1] for lower in confint(npe; level = 0.05)]
        upper_bounds = [upper[2] for upper in confint(npe; level = 0.05)]
        df = DataFrame(Sₑ = npe.Sₑ, ∂Λₑ = npe.∂Λₑ, σₑ=npe.σₑ, lower_95_CI = lower_bounds, upper_95_CI = upper_bounds)
        show(io, df)
    else
        print(io, "$(E)(t ∈ $(extrema(npe.grid)))")
    end
end


end
