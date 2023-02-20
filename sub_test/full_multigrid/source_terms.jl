using SparseArrays
using LinearAlgebra


σ = 0

function exact_u(C,k,σ,x)
    return C/(π^2*k^2 + σ) * sin.(k*π*x)
end

function source_terms(C,k,x)
    return C*sin.(k*π*x)
end

