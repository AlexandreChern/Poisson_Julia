# Solving the Poisson Equation
# We are solving the following Poisson Equation in 1D
#
# $$\left\{\begin{array}{rlrl}{u_{xx}} + \pi^2sin(\pi x) & {=0,} & {0\leq x \leq 1}  \\
# {u_{0}} & {=0,} & {x=0} \\     Dirichlet Boundary Condition
# {u}_x_{|1} & {=-\pi,} & {x=1}\end{array}\right.$$                  Neumann Boundary Condition

# Using external Julia file

include("diagonal_sbp.jl")
using Plots

using LinearAlgebra
using SparseArrays
function e(i,n)
    A = Matrix{Float64}(I,n,n)
    return A[:,i]
end

function eyes(n)
    return Matrix{Float64}(I,n,n)
end


function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end

# Solve for special case h = 0.05, n = 20
# Generating data

n_list = Array{Int64,1}(undef,10)
for i in range(1,step=1,stop=10)
    n_list[i] = Integer(3)^(i+1)
end

h_list = 1 ./ n_list

function Operators_2d(i, j, p=2, h_list_x = h_list, h_list_y = h_list)
    hx = h_list_x[i];
    hy = h_list_y[j];

    x = range(0,step=hx,1);
    y = range(0,step=hy,1);
    m_list = 1 ./h_list_x;
    n_list = 1 ./h_list_y;

    # Matrix Size
    n_x = Integer(m_list[i]);
    n_y = Integer(n_list[j]);

    N_x = n_x + 1
    N_y = n_y + 1

    n_x_one_third = Integer(n_x/3)
    n_y_one_third = Integer(n_y/3)

    N_x_one_third = n_x_one_third + 1
    N_y_one_third = n_x_one_third + 1

    (D1x, HIx, H1x, r1x) = diagonal_sbp_D1(p,n_x_one_third,xc=(0,1/3));
    (D2x, S0x, SNx, HI2x, H2x, r2x) = diagonal_sbp_D2(p,n_y_one_third,xc=(0,1/3));


    (D1y, HIy, H1y, r1y) = diagonal_sbp_D1(p,n_x_one_third,xc=(0,1/3));
    (D2y, S0y, SNy, HI2y, H2y, r2y) = diagonal_sbp_D2(p,n_y_one_third,xc=(0,1/3));

    BSx = sparse(SNx - S0x);
    BSy = sparse(SNy - S0y);


    # Forming 2d Operators
    e_1x = sparse(e(1,N_x+1));
    e_Nx = sparse(e(N_x+1,N_x+1));
    e_1y = sparse(e(1,N_y+1));
    e_Ny = sparse(e(N_y+1,N_y+1));


    I_Nx = sparse(eyes(N_x+1));
    I_Ny = sparse(eyes(N_y+1));


    e_E = kron(e_Nx,I_Ny);
    e_W = kron(e_1x,I_Ny);
    e_S = kron(I_Nx,e_1y);
    e_N = kron(I_Nx,e_Ny);

    E_E = kron(sparse(Diag(e_Nx)),I_Ny);   # E_E = e_E * e_E'
    E_W = kron(sparse(Diag(e_1x)),I_Ny);
    E_S = kron(I_Nx,sparse(Diag(e_1y)));
    E_N = sparse(kron(I_Nx,sparse(Diag(e_Ny))));


    D1_x = kron(D1x,I_Ny);
    D1_y = kron(I_Nx,D1y);


    D2_x = kron(D2x,I_Ny);
    D2_y = kron(I_Nx,D2y);
    D2 = D2_x + D2_y


    HI_x = kron(HIx,I_Ny);
    HI_y = kron(I_Nx,HIy);

    H_x = kron(H1x,I_Ny);
    H_y = kron(I_Nx,H1y);

    BS_x = kron(BSx,I_Ny);
    BS_y = kron(I_Nx,BSy);


    HI_tilde = kron(HIx,HIx);
    H_tilde = kron(H1x,H1y);

    return (D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N)
end

# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
# n_list = Int(1 ./h_list)


p = 2

i = 5

h = h_list[i]

n = Integer(n_list[i])
n_one_third = Integer(n/3)
N = n + 1
N_one_third = n_one_third + 1

(D1, HI1, H1, r1) = diagonal_sbp_D1(p,n_one_third,xc=(0,1/3)) # be careful about domain
(D2, S0, SN, HI2, H2, r2) = diagonal_sbp_D2(p,n_one_third,xc=(0,1/3))

#span = LinRange(0,1,N)
#analy_sol = sin.(span*π)

span_1 = LinRange(0,1/3,N_one_third)
span_2 = LinRange(1/3,2/3,N_one_third)
span_3 = LinRange(2/3,1,N_one_third)
span= vcat(span_1,span_2,span_3)
analy_sol = vcat(sin.(span_1*π),sin.(span_2*π),sin.(span_3*π))

e0 = e(1,N_one_third);
en = e(N_one_third,N_one_third);
BS = SN - S0

α = Dict(2=>1, 4=>0.2508560249, 6=>0.1878715026)
γ = 2 # γ > 1 for stability
#σ₁ = -γ/(α[p]*h_list[i]) # For left boundary Dirichlet condition
#σ₂ = -γ/(α[p]*h_list[i]) # For right boundary Neumann condition
σ₁ = -40
σ₂ = 1
β = 1
ϵ = 1  # Intersection

g_L = 0
g_R = -π



# NOW DEFINING HYBRID METHOD parameters
F_L = -π^2*sin.(span_1*π)
F_M = -π^2*sin.(span_2*π)
F_R = -π^2*sin.(span_3*π)

L1 = e0'
L2 = en'
τ = σ₁
δ_f = 0.1

# G1 = L1*BS*h
# G2 = L2*BS*h

g_L = 0
g_R = -π



Mu =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + τ*L2'*L2 + β*BS'*L2'*L2
Mv =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + τ*L2'*L2 + β*BS'*L2'*L2
Mw =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + β*L2'*L2*BS + 1/τ*BS'*L2'*L2*BS#*BS #+G1

# Mw = H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + β*L2'*L2*BS + 1/τ*BS'*L2'*L2*BS

# F = vcat(τ*L2'+BS'*L2', τ*L1' + BS'*L1' + τ*L2' + BS'*L2' ,τ*L1'+ BS'*L1')
# F_T = hcat(τ*L2+L2*BS, τ*L1 + L1*BS + τ*L2 + L2*BS  ,τ*L1 + L1*BS )
b_zero = zeros(N_one_third)

F = hcat(vcat(τ*L2'+BS'*L2',τ*L1'+ BS'*L1',b_zero),vcat(b_zero,τ*L2' + BS'*L2', τ*L1' + BS'*L1'))
F_T = vcat(hcat(τ*L2+L2*BS, τ*L1 + L1*BS , b_zero'),hcat(b_zero', τ*L2 + L2*BS, τ*L1 + L1*BS))

# D = 4*τ*h
D = vcat(hcat(2*τ,0),hcat(0,2*τ))
# D = vcat(hcat(τ,τ),hcat(τ,τ))

g_bar = vcat(τ*L1'*g_L + BS'*L1'*g_L + H1*F_L, H1*F_M ,L2'*g_R + 1/τ*BS'*L2'*g_R + H1*F_R)
# g_bar_delta = 2*h*δ_f
g_bar_delta = vcat(2*h*δ_f,2*h*δ_f) # Not Sure

Mzero = zeros(N_one_third,N_one_third)

A1 = vcat(hcat(Mu,Mzero,Mzero),hcat(Mzero,Mv,Mzero),hcat(Mzero,Mzero,Mw))

A = vcat(hcat(A1,F),hcat(F_T,D))
b = vcat(g_bar,g_bar_delta)

num_sol_2 = A\b
num_sol_2_tranc = num_sol_2[1:end-2]
plot(span,num_sol_2_tranc)
