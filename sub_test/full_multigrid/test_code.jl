include("smoothers.jl")
include("source_terms.jl")
include("interpolations.jl")

using Random
Random.seed!(0)

## Testing smoothers
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
ω = 2/3

x_1 = ω * D\(b - LU*x) + (1-ω) * x
x_2 = ω * D\(b - LU*x_1) + (1-ω) * x_1
x_3 = ω * D\(b - LU*x_2) + (1-ω) * x_2
x_4 = ω * D\(b - LU*x_3) + (1-ω) * x_3
x_5 = ω * D\(b - LU*x_4) + (1-ω) * x_4

Jacobi_iter(A,b,x;nu=5)  

@assert x ≈ x_5

# Finished testing Jacobi_iter

## Testing A_matrix function and retesting smoothers
A = A_matrix(63)
b = randn(size(A,1))
D = Diagonal(A)
LU = A - D
x = zeros(size(b))
# ω = 2/3
ω = 1
x_1 = ω * D\(b - LU*x) + (1-ω) * x
x_2 = ω * D\(b - LU*x_1) + (1-ω) * x_1
x_3 = ω * D\(b - LU*x_2) + (1-ω) * x_2
x_4 = ω * D\(b - LU*x_3) + (1-ω) * x_3
x_5 = ω * D\(b - LU*x_4) + (1-ω) * x_4

Jacobi_iter(A,b,x;nu=5,ω=ω)  
@assert x ≈ x_5


## Testing interpolation and restriction

N = 2^6
A = A_matrix(N-1)
x_range = range(0,1,step=1/N)[2:end-1]
b = source_terms(C,k,x_range)
x = zeros(size(b))

A_coarse = restrict_matrix(A)
b_coarse = weighting(b)

A_coarse \ b_coarse

## Finished testing interpolation and restriction


## Testing V_cycle
MG_sol = V_cycle(A,b,x;levels=3,iter_times=3)
analytical_sol = exact_u(C,k,σ,x_range)

error_norm = 1/N^2 * norm(MG_sol - analytical_sol)
## Finished testing V_cycle

