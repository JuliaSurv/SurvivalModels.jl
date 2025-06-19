### This file is suppose to produce all extra tests to ensure that our implementations are behaving correclty. 

### It should not do anything else, and no other julia file than main.jl and tests.jl should be main entry points. Launching these two files should be enough to reproduce all the results. 

const ovarian = let
    df = dataset("survival","ovarian")
    T = df.FUTime
    T[11] = T[10]
    T[9] = T[10]
    Δ = df.FUStat
    X = [df.Age Float64.(df.ECOG_PS)]
    (T,Δ,X)
end

const colon = let
    df = dataset("survival","colon")
    df = dropmissing(df, [:Nodes, :Differ])
    
    T = Float64.(df.Time)
    Δ = Int64.(df.Status)
    X = Float64.(hcat(df.Sex, df.Age, df.Obstruct,
    df.Perfor, df.Adhere, df.Nodes,
    df.Differ, df.Extent, df.Surg,
    df.Node4, df.EType))
    (T,Δ,X)
end

# This file is supposed to contain testing functions to esnure correct behavior of the code. 
# It should be independent of the main file 
# and it should simply load the different version and check on both colon and ovarian that they give the right results... 
ovR = getβ(CoxVR(ovarian...))
clR = getβ(CoxVR(colon...))

@testitem "Resultats" begin
    const design = Dict(
    # Label => (constructor, plotting color)
    "R" => (CoxVR, :red),
    "Jl"=> (CoxVJ, :blue),
    "V1"=> (CoxV1, :orange),
    "V2"=> (CoxV2, :brown),
    "V3"=> (CoxV3, :purple),
    "V4"=> (CoxV4, :green),
    "V5"=> (CoxV5, :black)
    );
    function simulate_survival_data(n, m; censor_rate = 0.2, β=randn(m))
        Random.seed!(42)
        X = hcat(
            [randn(n)       for _ in 1:cld(m,3)]..., # about 1/3
            [rand(n)        for _ in 1:cld(m,3)]..., # about 1/3
            [exp.(randn(n)) for _ in 1:(m-2cld(m,3))]... # the rest. 
        )
        η = X * β
        λ₀ = 1 # rand(Exponential(1.0), n) <<<------- This was not a Cox model before, which is why the beta were not recovered correctly i think.
        U = rand(n)
        O = -log.(U) ./ (λ₀ .* exp.(η))
        C = rand(Exponential(quantile(O, 1 - censor_rate)), n) #### <- The censor rate was reversed here, changed it for readibility. 
        T = min.(O, C)
        Δ = Bool.(T .<= C)
        return (T, Δ, X)
    end
    function run_models() 
        Ns = (500, 1000, 5000) # 10_000, 100_000
        Ms = (10, 20, 50) #100
        true_betas = randn(maximum(Ms))
        cond(n,m) = (n==5000)||(m==20)
        function cond(name, n, m)
            cond1 = (n==5000)||(m==20)
            cond2 = (name ∉ ("V1","V2")) || ((n <= 1000) && (m <=20))
            return cond1 & cond2
        end
        df = []
        for n in Ns, m in Ms
            if cond(n,m)
                data = simulate_survival_data(n,m, β = true_betas[1:m])
                for (name, (constructor,color)) in design
                    if cond(name, n,m)
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
                            diff_to_truth = L2dist(beta, true_betas[1:m]) / norm(true_betas[1:m]),
                        ))
                    end
                end
            end
        end
        df = DataFrame(df)
        sort!(df, :name)
        return df
    end
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
    
        group2 = groupby(filter(r -> r.n==5000, df), :name)
        p2 = plot(; xlabel = "Number of covariates (m)",
                    ylabel = "Temps (ms)",
                    yscale= :log10,
                    xscale= :log10,
                    title = "For n=5000 obs., varying m",
                    legend = :bottomright,
                    lw = 1);
        for g in group2
            plot!(p2, g.m, g.time, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
        end
        p = plot(p1,p2, size=(1200,600), plot_title = "Runtime (logscale) of the various implementations")
        return p
    end
    function beta_correctness_graphs(df; ref="R")
    
        reflines = filter(r -> r.name == ref, df)
        rename!(reflines, :beta => :refbeta)
        select!(reflines, Not([:name, :time,:diff_to_truth]))
        otherlines = filter(r -> r.name != ref, df)
        rez = leftjoin(otherlines, reflines, on=[:n,:m])
        rez.error = L2dist.(rez.beta, rez.refbeta) ./ norm.(rez.refbeta)
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
    
        group2 = groupby(filter(r -> r.n==5000, rez), :name)
        p2 = plot(; xlabel = "Nomber of covariates (m)",
                    ylabel = "L2Dist to $ref's β",
                    yscale=:log10,
                    xscale= :log10,
                    title = "n=5000, varying m",
                    legend = :bottomright,
                    lw = 1);
        for g in group2
            plot!(p2, g.m, g.error, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
        end
        p = plot(p1,p2, size=(1200,600), plot_title="β-correctness w.r.t. $ref's version.")
        return p
    end
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
    
        group2 = groupby(filter(r -> r.n==5000, df), :name)
        p2 = plot(; xlabel = "Nomber of covariates (m)",
                    ylabel = "L2Dist to the truth",
                    yscale=:log10,
                    xscale= :log10,
                    title = "n=5000, varying m",
                    legend = :bottomright,
                    lw = 1);
        for g in group2
            plot!(p2, g.m, g.diff_to_truth, label = g.name[1] , color = design[g.name[1]][2], marker = :circle, markersize = 3)  
        end
        p = plot(p1,p2, size=(1200,600), plot_title="β-correctness w.r.t. the truth.")
        return p
    end
    
    df = run_models()
    CSV.write("results2.csv", df)
    savefig(timing_graph(df), "out/timings.pdf")
    savefig(beta_correctness_graphs(df, ref="R"), "out/beta_wrt_R(small).pdf")
    savefig(beta_correctness_graphs(df, ref="Jl"), "out/beta_wrt_Jl(small).pdf")
    savefig(beta_wrt_truth(df), "out/beta_wrt_truth(small).pdf")

end


