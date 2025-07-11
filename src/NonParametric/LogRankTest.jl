"""
    LogRankTest(T, Δ, group, strata)
    fit(LogRankTest, @formula(Surv(T, Δ) ~ gr), data = ...)
    fit(LogRankTest, @formula(Surv(T, Δ) ~ Strata(st) + gr), data = ...)

Performs the stratified log-rank test for comparing survival distributions across groups.

# Arguments
- `T`: Vector of observed times.
- `Δ`: Vector of event indicators (`1` = event, `0` = censored).
- `group`: Vector indicating group membership (e.g., treatment arm).
- `strata`: Vector indicating strata membership (e.g., baseline strata).
- `gr` and `st` are the variables in the DataFrame defining the groups and strata for the `fit` interface.

# Returns
A `LogRankTest` object with the following fields:
- `stat`: Chi-square test statistic.
- `df`: Degrees of freedom (number of groups minus 1).
- `pval`: P-value of the test.

# Notes
- Implements the stratified log-rank test by aggregating test statistics and variances over strata.
- Suitable for right-censored survival data with stratification.
"""
struct LogRankTest{T}
    ∂N::Array{T, 3}
    ∂V::Array{T, 3}
    ∂Z::Array{T, 3}
    D::Array{T, 3}
    R::Array{T, 3}
    ∂VZ::Array{T, 4}
    stat::Float64
    df::Int
    pval::Float64
    function LogRankTest(Times, Δ, group, strata)
        T0 = eltype(Times)
        T = promote_type(T0, Float64)
        grid = sort(unique(Times))
        stratas = unique(strata)
        groups  = unique(group)
        nstrata = length(stratas)
        ngroups = length(groups)
        ngrid = length(grid)

        ∂N  = zeros(T, nstrata, ngroups, ngrid)
        ∂V  = zeros(T, nstrata, ngroups, ngrid)
        ∂Z  = zeros(T, nstrata, ngroups, ngrid)
        D   = zeros(T, nstrata, ngroups, ngrid)
        R   = zeros(T, nstrata, ngroups, ngrid)
        ∂VZ = zeros(T, nstrata, ngroups, ngroups, ngrid)

        for s in eachindex(stratas)
            for g in eachindex(groups)
                idx = Bool.((group .== groups[g]) .* (strata .== stratas[s]))
                model = KaplanMeier(Times[idx], Δ[idx])
                ∂N[s, g, :], ∂V[s, g, :], D[s, g, :] = T.(model.∂N), T.(model.∂N) ./ (T.(model.∂N) .- T.(model.Y)), T.(model.Y)
            end
        end

        R .= ifelse.(sum(D,dims=2) .== 0, zero(T), D ./ sum(D,dims=2))
        ∂Z .= ∂N .- R .* sum(∂N,dims=2)

        for s in eachindex(stratas)
            for ℓ in eachindex(groups)
                for g in eachindex(groups)
                    for h in eachindex(groups)
                        for t in eachindex(grid)
                            ∂VZ[s, g, h, t] += ((g==ℓ) - R[s, g, t]) * ((h==ℓ) - R[s, h, t]) * ∂V[s, ℓ, t]
                        end
                    end
                end
            end
        end

        Z =  dropdims(sum(∂Z, dims=(1,3)), dims=(1,3))
        VZ = dropdims(sum(∂VZ, dims=(1,4)), dims=(1,4))

        stat = dot(Z[1:end-1], (VZ[1:end-1,1:end-1] \ Z[1:end-1]))
        df = ngroups-1
        pval = ccdf(Chisq(df), stat[1])
        return new{T}(∂N, ∂V, ∂Z, D, R, ∂VZ, stat[1], df, pval)
    end
end



# The fitting and formula interfaces should be here. 
function StatsBase.fit(::Type{E}, formula::FormulaTerm, df::DataFrame) where {E<:LogRankTest}

    terms = StatsModels.termvars(formula.rhs)
    tf = typeof(formula.rhs)
    types = (tf <: AbstractTerm) ? [tf] : typeof.(formula.rhs)
    are_strata = [t <: FunctionTerm{typeof(Strata)} for t in types]

    strata = groupindices(groupby(df,terms[are_strata]))
    group  = groupindices(groupby(df,terms))
    resp = modelcols(apply_schema(formula,schema(df)).lhs,df)
    
    return LogRankTest(resp[:,1], resp[:,2], group, strata)
end
