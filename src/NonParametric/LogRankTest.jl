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
        nstrata, ngroups, ngrid = length(stratas), length(groups), length(grid)

        ∂N  = zeros(T, nstrata, ngroups, ngrid)
        D   = zeros(T, nstrata, ngroups, ngrid)
        ∂V  = zeros(T, nstrata, ngroups, ngrid)
        ∂Z  = zeros(T, nstrata, ngroups, ngrid)
        R   = zeros(T, nstrata, ngroups, ngrid)
        ∂VZ = zeros(T, nstrata, ngroups, ngroups, ngrid)

        # Precompute masks for each (stratum, group)
        masks = [((strata .== stratum_val) .& (group .== group_val)) for stratum_val in stratas, group_val in groups]

        # Main loop: for each event time, fill D and ∂N for all strata and groups
        for (j, tj) in enumerate(grid)
            for s in 1:nstrata, g in 1:ngroups
                mask = masks[s, g]
                T_gs = Times[mask]
                Δ_gs = Δ[mask]
                D[s, g, j] = sum(T_gs .>= tj)
                ∂N[s, g, j] = sum((T_gs .== tj) .& (Δ_gs .== 1))
            end
        end

        # Now, for each stratum and time, compute totals and fill R, ∂Z, ∂V
        for s in 1:nstrata, j in 1:ngrid
            Dtot = sum(D[s, :, j])
            ∂Ntot = sum(∂N[s, :, j])
            for g in 1:ngroups
                R[s, g, j] = D[s, g, j] > 0 ? D[s, g, j] / Dtot : zero(T)
                ∂Z[s, g, j] = ∂N[s, g, j] - R[s, g, j] * ∂Ntot
                ∂V[s, g, j] = (Dtot > 1) ? (∂Ntot * D[s, g, j] * (Dtot - D[s, g, j]) / (Dtot^2 * (Dtot - 1))) : zero(T)
            end
            # Fill ∂VZ for this stratum and time
            for ℓ in 1:ngroups, g in 1:ngroups, h in 1:ngroups
                ∂VZ[s, g, h, j] += ((g==ℓ) - R[s, g, j]) * ((h==ℓ) - R[s, h, j]) * ∂V[s, ℓ, j]
            end
        end

        Z = dropdims(sum(∂Z, dims=(1,3)), dims=(1,3))
        VZ = dropdims(sum(∂VZ, dims=(1,4)), dims=(1,4))

        stat = dot(Z[1:end-1], (VZ[1:end-1,1:end-1] \ Z[1:end-1]))
        df = ngroups-1
        pval = ccdf(Chisq(df), stat)
        return new{T}(∂N, ∂V, ∂Z, D, R, ∂VZ, stat, df, pval)
    end
end

# The fitting and formula interfaces should be here. 
function StatsBase.fit(::Type{E}, formula::FormulaTerm, df::DataFrame) where {E<:LogRankTest}
    terms = StatsModels.termvars(formula.rhs)
    tf = typeof(formula.rhs)
    types = (tf <: AbstractTerm) ? [tf] : typeof.(formula.rhs)
    are_strata = [t <: FunctionTerm{typeof(Strata)} for t in types]

    # Separate group and strata terms
    strata_terms = terms[are_strata]
    group_terms = terms[.!are_strata]

    # Compute group and strata indices
    strata = isempty(strata_terms) ? ones(Int, nrow(df)) : groupindices(groupby(df, strata_terms))
    group  = isempty(group_terms)  ? ones(Int, nrow(df)) : groupindices(groupby(df, group_terms))

    resp = modelcols(apply_schema(formula, schema(df)).lhs, df)
    return LogRankTest(resp[:,1], resp[:,2], group, strata)
end
