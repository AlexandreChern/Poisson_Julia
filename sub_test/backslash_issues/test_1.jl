using SparseArrays
using LinearAlgebra

n = 5

A = randn(n,n)
b = randn(n)

F = randn(n,2)

A_sparse = sparse(A)
F_sparse = sparse(F)

LU_A = lu(A_sparse)

LU_A.L
LU_A.U


LU_A.Rs
LU_A.L*LU_A.U

@show LU_A.L*LU_A.U

@show LU_A.Rs .* A[LU_A.p,LU_A.q]

#1 direct method
@show A\F


#2 LU decomposition form 1
LU_A.Rs .\ LU_A.L*LU_A.U \ F

#3 LU decomposition form 2
LU_A.L*LU_A.U \ (LU_A.Rs .* F)

#3.1 LU decomposition another form
LU_A.U\(LU_A.L\(LU_A.Rs .* F))

#4 similar to form 1 with sparse F,  failed!
LU_A.Rs .\ LU_A.L*LU_A.U \ F_sparse

#5 similar to form 2 with sparse F, still failed
LU_A.L*LU_A.U \(LU_A.Rs .* F_sparse)

#6 looks like this works, yep!
LU_A.U\(LU_A.L\(Diagonal(LU_A.Rs)*F_sparse))

#7 test, also works!
LU_A.U\(LU_A.L\(LU_A.Rs .* F_sparse))

Diagonal(LU_A.Rs)*F_sparse
LU_A.Rs .* F_sparse



# Correct Form that works with general p and q

X = zeros(eltype(B),size(B))
X[LU_A.q,:] = LU_A.U\(LU_A.L\((Diagonal(LU_A.Rs) * B)[LU_A.p,:]))






# Equivalent form with wrong precedence but works
LU_A.L\F
LU_A.L\F_sparse





LU_A.U\F
LU_A.U\F_sparse

LU_A.L\(LU_A.U\F)
LU_A.L\(LU_A.U\F_sparse)

LU_A.U\(LU_A.L\F)
LU_A.U\(LU_A.L\F_sparse)

tmp1 = LU_A.U\(LU_A.L\F)
tmp2 = LU_A.U\(LU_A.L\F_sparse)

LU_A.Rs .\tmp1

LU_A.Rs .\tmp2
# these two are equivalent, but not equivalent to

LU_A.Rs .\ LU_A.L*LU_A.U \F

# which is the correct solution to A\F
