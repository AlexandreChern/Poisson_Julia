using SparseArrays
using LinearAlgebra


σ = 0

function exact_u(C,k,σ,x)
    return C/(π^2*k^2 + σ) * sin.(k*π*x)
end

function source_terms(C,k,x)
    return C*sin.(k*π*x)
end


function A_matrix(N)
    A = spzeros(N,N)
    h = 1 / (N+1)
    for i in 1:N
        A[i,i] = 2
    end

    for i in 1:N-1
        A[i,i+1] = -1
        A[i+1,i] = -1
    end
    return A ./ h^2
end