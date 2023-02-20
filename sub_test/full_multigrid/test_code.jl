include("smoothers.jl")

using Random
Random.seed!(0)

# Testing smoothers
# Generate a random 5 by 5 matrix
A = randn(5,5)
A = A*A' + 2*sparse(I,5,5) # Create a positive definite matrix

# Generate right-hand-side
b = randn(size(A,1))

# split matrices
D = Diagonal(A)
LU = A - D

# Jacobi interpolation
x = zeros(size(b))

x_1 = D\(b - LU*x)
x_2 = D\(b - LU*x_1)
x_3 = D\(b - LU*x_2)
x_4 = D\(b - LU*x_3)
x_5 = D\(b - LU*x_4)

Jacobi_iter(A,b,x;nu=5)  

@assert x ≈ x_5

# Finished testing Jacobi_iter