struct CoxVJ
    T::Vector{Float64}
    Δ::Vector{Bool}
    X::Matrix{Float64}
    function CoxVJ(T,Δ,X)
        new(T,Bool.(Δ),X)
    end
end

function getβ(M::CoxVJ)
    return fit(Survival.CoxModel, M.X, Survival.EventTime.(M.T,M.Δ)).β
end


