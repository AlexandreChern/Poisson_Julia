using LinearAlgebra
using IterativeSolvers

struct SizedStrangMatrix
    size::Tuple{Int, Int}
end

Base.eltype(A::SizedStrangMatrix) = Float64
Base.size(A::SizedStrangMatrix) = A.size
Base.size(A::SizedStrangMatrix,i::Int) = A.size[i]

B = rand(10)
A = SizedStrangMatrix((length(B),length(B)))

#
# function Base.A_mul_B!(C,A::SizedStrangMatrix,B)  # A_mul_B! is depricated use LinearAlgebra.mul!() instead
#     for i in 2:length(B) - 1
#         C[i] = B[i-1] - 2B[i] + B[i+1]
#     end
#     C[1] = -2B[1] + B[2]
#     C[end] = B[end-1] - 2B[end]
#     C
# end

function LinearAlgebra.mul!(C,A::SizedStrangMatrix,B)
    for i in 2:length(B) - 1
        C[i] = B[i-1] - 2B[i] + B[i+1]
    end
    C[1] = -2B[1] + B[2]
    C[end] = B[end-1] - 2B[end]
    C
end

Base.:*(A::SizedStrangMatrix,B::AbstractVector) = (C = similar(B); mul!(C,A,B))


# A = rand(10,10)




U = gmres(A,B,reltol=1e-14)
norm(A*U - B)
