using LinearAlgebra
using SparseArrays


n = 10
b = randn(n)


A = randn(n,n)
A_s = sparse(A)

F = lu(A_s)

x = similar(b)
x[F.q,:] = F.U\sparse(F.L\(F.Rs .* b)[F.p,:])

@show x
@show A\b
