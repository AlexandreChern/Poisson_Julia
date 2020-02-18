using SparseArrays
using LinearAlgebra
using BenchmarkTools


N = 20 # dimension of SparseArrays
A = randn(N,N);
F = randn(N,10);

sparse_A = sparse(A)
sparse_F = sparse(F)

@show typeof(A)
@show typeof(F)
@show typeof(sparse_A)
@show typeof(sparse_F)

size(sparse_F,2)

function backslash_for_sparse(A,F)
    dim_n = size(F,2)
    output = similar(F)
    tmp = similar(F)
    lu_A = lu(A)
    @inbounds for i = 1:dim_n
        tmp[:,i] = @view F[:,i]
        # output[:,i] = ldiv!(output[:,i],lu_A,Vector(tmp[:,i]))
        output[:,i] = ldiv!(lu_A,Vector(tmp[:,i]))
    end
    return output
end

@benchmark lu_A = lu(A)
lu_A = lu(A)
function backslash_for_sparse_v1(lu_A,F)
    output = similar(F)
    @inbounds for i=1:size(F,2)
        output[:,i] = ldiv!(lu_A,Vector(F[:,i]))
    end
    return output
end

A\F
backslash_for_sparse(A,F)
backslash_for_sparse(sparse_A,sparse_F)
backslash_for_sparse_v1(lu_A,F)
backslash_for_sparse_v1(lu_A,sparse_F)

@benchmark A\F
@benchmark backslash_for_sparse(A,F)
@benchmark backslash_for_sparse(sparse_A,sparse_F)
@benchmark lu(A)
@benchmark backslash_for_sparse_v1(lu_A,F)
@benchmark backslash_for_sparse_v1(lu_A,sparse_F)

using SuiteSparse
function myLUsolve(A,B)

    # get parts of LUfactorization
    lu_A = lu(A)
    println("time to extract LUfactors")
    @time (L,U,p,q,R) = (lu_A.L,lu_A.U,lu_A.p,lu_A.q,lu_A.Rs)

    println("time to solve for $(size(B,2)) right hand sides")
    @time begin
        X = zeros(eltype(B),size(B))
        X[q,:] = U\(L\((Diagonal(R) * B)[p,:]))
    end
    return X
end

nrhs = 300
n    = 10000
A    = UniformScaling(10) + sprandn(n,n,0.0005);
rhs  = randn(n,nrhs);

println("=== lufact(A)\\rhs for  real matrix ===")
println("\t factorize")
@time luA = lu(A)
println("\t julia solve")
@time t1 =  luA\rhs
println("\t my solve")
@time t2 = myLUsolve(luA,rhs)


println("err: $(norm(t1-t2)/norm(t1))")
