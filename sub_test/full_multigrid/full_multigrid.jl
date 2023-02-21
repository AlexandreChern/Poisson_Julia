using LinearAlgebra
using SparseArrays

include("source_terms.jl")
include("interpolations.jl")
include("smoothers.jl")


function V_cycle(A,b,x;levels=3,iter_times=3)
    v = x
    N = length(b)
    v_values = Dict(1=>v)
    rhs_values = Dict(1=>b)
    for i in 1:levels
            if i != levels
                for _ in 1:iter_times
                    Jacobi_iter(A,rhs_values[i],v_values[i])
                end
                # restrict RHS
                rhs = weighting(rhs_values[i] - A*(v_values[i]))
                rhs_values[i+1] = rhs
                N = div(N,2)
                v = zeros(N)
                v_values[i+1] = v
                # restrict Matrix
                A = restrict_matrix(A)
                @show size(v)
                @show size(A)
                @show size(rhs)
            else
                v_values[i] = A_matrix(N) \ rhs_values[i]
            end
    end
end