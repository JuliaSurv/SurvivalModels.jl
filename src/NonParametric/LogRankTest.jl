
struct LogRankTest
    ∂N::Array{Float64, 3}
    ∂V::Array{Float64, 3}
    ∂Z::Array{Float64, 3}
    D::Array{Float64, 3}
    R::Array{Float64, 3}
    ∂VZ::Array{Float64, 4}
    stat::Float64
    df::Int64
    pval::Float64
    function LogRankTest(T, Δ, group, strata)
        # This should be tested a bit more, not sure if it is true. 

        grid = sort(unique(T))

        # get stratas and groups, count them.  
        stratas = unique(strata)
        groups  = unique(group)
        nstrata = length(stratas)
        ngroups = length(groups)

        # Allocate: 
        ∂N  = zeros(nstrata, ngroups, length(grid))
        ∂V  = zeros(nstrata, ngroups, length(grid))
        ∂Z  = zeros(nstrata, ngroups, length(grid))
        D   = zeros(nstrata, ngroups, length(grid))
        R   = zeros(nstrata, ngroups, length(grid))
        ∂VZ = zeros(nstrata, ngroups, ngroups, length(grid))

        # Comput KM numerator and denominators on each strata&group (s,g)
        for s in eachindex(stratas)
            for g in eachindex(groups)
                idx = Bool.((group .== groups[g]) .* (strata .== stratas[s]))
                model = KaplanMeier(T[idx], Δ[idx])
                ∂N[s, g, :], ∂V[s, g, :], D[s, g, :] = model.d, model.d / (model.n - model.d), model.n
            end
        end

        # renormalize on groups, be carefull for zeros. 
        R .= ifelse.(sum(D,dims=2) .== 0, 0, D ./ sum(D,dims=2))
        ∂Z .= ∂N .- R .* sum(∂N,dims=2)
        
        # Compute test variance on each strata
        for s in eachindex(stratas)
            for ℓ in eachindex(groups)
                for g in eachindex(groups)
                    for h in eachindex(groups)
                        for t in eachindex(grid)
                            ∂VZ[s, g, h, t] += ((g==ℓ) - R[s, g, t]) * ((h==ℓ) - R[s, h, t]) .* ∂V[s, ℓ, t]
                        end
                    end
                end
            end
        end

        # Cumulate accross time and stratas
        Z =  dropdims(sum(∂Z, dims=(1,3)), dims=(1,3))
        VZ = dropdims(sum(∂VZ, dims=(1,4)), dims=(1,4))

        # Finally compute the stat and p-values:
        stat = dot(Z[1:end-1],(VZ[1:end-1,1:end-1] \ Z[1:end-1])) # test statistic
        df = ngroups-1 # number of degree of freedom of the chi-square test
        pval = ccdf(Chisq(df), stat[1]) # Obtained p-value. 
        return new(∂N, ∂V, ∂Z, D, R, ∂VZ, stat[1], df, pval)
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
