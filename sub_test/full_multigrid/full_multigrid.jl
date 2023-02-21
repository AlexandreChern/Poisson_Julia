using LinearAlgebra
using SparseArrays

include("source_terms.jl")
include("interpolations.jl")
include("smoothers.jl")


function V_cycle(A,b,x;levels=3)
    f = b
    for i in levels:-1:1
        if i == 1:
            x = A\f
        else

        end
    end
end