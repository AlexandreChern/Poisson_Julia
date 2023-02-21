using SparseArrays
using LinearAlgebra


# function Jacobi_iter(ω,v,f)
#     N = length(v)
#     h = 1/(N+1)
#     v_new = copy(v)
#     for j = 2:N-1
#         v_new[j] = (1-ω) * v[j] + ω * 1/(2 + σ*h^2) * (v[j-1] + v[j+1] + h^2*f[j])
#     end
#     return v_new
# end

function Jacobi_iter(A,b,x;nu=10,ω = 1)
    D = Diagonal(A)
    LU = A - D
    for i in 1:nu
        x .= ω * D\(b - LU*x) .+ (1-ω) * x
    end
end

