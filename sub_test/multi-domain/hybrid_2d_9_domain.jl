# Solving the Poisson Equation
# We are solving the following Poisson Equation in 2D
# Dirichlet Boundary Condition on West and East
# Neumann Boundary Condition on North and South
# Δu = -2π^2*u
# Manufactured Solution would be u(x,y) = sin(πx + πy)
# u(0,y) = sin(y)                   // Dirichlet Boundary Condition on West Side
# u(1,y) = sin(π+y) = -sin(y)       // Dirichlet Boundary Condition on East Side
# ∂u(x,0)/∂y = π*cos(π*x)           // Neumann Boundary Condition on South Side
# ∂u(x,1)/̡∂y = -π*cos(π*x)          // Neumann Boundary Condition on North Side

# Using external Julia file


# We split our domain into the following 9 blocks

# L M B for left Middle Right
# B M T for Bottom Middle Top

# For each block
# W E S T for West East South North

# Eg: We refer the interfect between LB and MB (of LB) as LB_E
# Eg: We refer the interfect between LB and MB (of LM) as MB_W
# Eg: We refer the interface between MM and MB (of MM) as MM_S
# Eg: We refer the interface between MM and MB (of MB) as MB_N

# All the boundary in this diagram
# LB_W, LB_S
# LM_W
# LT_W, LT_N
#
# MB_S
#
# MT_N
#
# RB_E, RB_S
# RM_E
# RT_E, RT_N

# All the interfaces in this diagram
# LB_E, LB_N
# LM_S, LM_E, b_LM
# LT_S, LT_E
#
# MB_W, MB_E, MB_N
# MM_W, MM_E, MM_S, MM_N
# MT_W, MT_E, MT_S
#
# RB_W,  RB_N
# RM_W, RM_S, RM_N
# RT_W, RT_s

# The number of boundaries is the complementary of the number of interfaces for each block. i.e sum = 4 for 2d



##############################
#    _____________________   #
#   |      |      |      |   #
#   |  LT  |  MT  |  RT  |   #
#   |______|______|______|   #
#   |      |      |      |   #
#   |  LM  |  MM  |  RM  |   #
#   |______|______|______|   #
#   |      |      |      |   #
#   |  LB  |  MB  |  RB  |   #
#   |______|______|______|   #
#                            #
##############################


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


function analy_sol(x,y) # Defines analytical_solution
    return sin.(π*x .+ π*y)
end


function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end



n_list = Array{Int64,1}(undef,6)
for i in range(1,step=1,stop=6)
    n_list[i] = Integer(3)^(i+1)
end

m_list = n_list

h_list = 1 ./ n_list

# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
# n_list = Int(1 ./h_list)

p = 2
i = j = 3

h = h_list[i]

n = Integer(n_list[i])
n_one_third = Integer(n)
N = n + 1
N_one_third = n_one_third + 1

function Operators_2d(i, j)
    h_list_x = h_list;
    h_list_y = h_list;

    hx = h_list_x[i];
    hy = h_list_y[j];

    n_x = Integer(n_list[i])
    n_y = Integer(m_list[j])

    # Matrix Size
    # n_x = Integer(m_list[i]);
    # n_y = Integer(n_list[j]);

    N_x = n_x + 1
    N_y = n_y + 1

    n_x_one_third = n_x      # Integer(n_x)
    n_y_one_third = n_y      # Integer(n_y)

    N_x_one_third = n_x_one_third + 1
    N_y_one_third = n_x_one_third + 1

    (D1x, HIx, H1x, r1x) = diagonal_sbp_D1(p,n_x_one_third,xc=(0,1/3));
    (D2x, S0x, SNx, HI2x, H2x, r2x) = diagonal_sbp_D2(p,n_y_one_third,xc=(0,1/3));


    (D1y, HIy, H1y, r1y) = diagonal_sbp_D1(p,n_x_one_third,xc=(0,1/3));
    (D2y, S0y, SNy, HI2y, H2y, r2y) = diagonal_sbp_D2(p,n_y_one_third,xc=(0,1/3));

    BSx = sparse(SNx - S0x);
    BSy = sparse(SNy - S0y);


    # Forming 2d Operators
    e_1x = sparse(e(1,N_x_one_third));
    e_Nx = sparse(e(N_x_one_third,N_y_one_third));
    e_1y = sparse(e(1,N_y_one_third));
    e_Ny = sparse(e(N_y_one_third,N_y_one_third));


    I_Nx = sparse(eyes(N_x_one_third));
    I_Ny = sparse(eyes(N_y_one_third));


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

i = j = 2
p = 2

h = h_list[i]
n = Integer(n_list[i])
N = n + 1
n_one_third = Integer(n)
N_one_third = n_one_third + 1

(D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j)

D2_x
E_E

#
# (D1, HI1, H1, r1) = diagonal_sbp_D1(p,n_one_third,xc=(0,1/3)) # be careful about domain
# (D2, S0, SN, HI2, H2, r2) = diagonal_sbp_D2(p,n_one_third,xc=(0,1/3))
#
# #span = LinRange(0,1,N)
#analy_sol = sin.(span*π)

span_1 = LinRange(0,1/3,N_one_third)
span_2 = LinRange(1/3,2/3,N_one_third)
span_3 = LinRange(2/3,1,N_one_third)
span= vcat(span_1,span_2,span_3)
analy_solution = analy_sol(span,span')
plot(span,span,analy_solution,st=:surface)


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

# L M R for Left Middle Right
# B M T for Bottom Middle Top
F_LB = -π^2*analy_sol(span_1,span_1')
F_LM = -π^2*analy_sol(span_1,span_2')
F_LT = -π^2*analy_sol(span_1,span_3')


F_MB = -π^2*analy_sol(span_2,span_1')
F_MM = -π^2*analy_sol(span_2,span_2')
F_MT = -π^2*analy_sol(span_2,span_3')

F_RB = -π^2*analy_sol(span_3,span_1')
F_RM = -π^2*analy_sol(span_3,span_2')
F_RT = -π^2*analy_sol(span_3,span_3')

## Reference Definition in 1D case
L1 = e0'
L2 = en'
τ = σ₁
δ_f = 0.1

# G1 = L1*BS*h
# G2 = L2*BS*h

# g_L = 0
# g_R = -π

# Boundary Conditions

# L M R for Left Middle Right
# B M T for Bottom Middle Top

g_W = sin.(π*span)  # Boundary conditions on the west side
g_E = -sin.(π*span) # Boundary conditions on the east side

g_S = π*cos.(π*span) # Boundary conditions on the south side
g_N = -π*cos.(π*span) # Bondary conditions on the north side

g_LB_W = g_W[1:N_one_third]
g_LM_W = g_W[N_one_third+1:2*N_one_third]
g_LT_W = g_W[2*N_one_third+1:3*N_one_third]

g_RB_E = g_E[1:N_one_third]
g_RM_E = g_E[N_one_third+1:2*N_one_third]
g_RT_E = g_E[2*N_one_third+1:3*N_one_third]

g_LB_S = g_S[1:N_one_third]
g_MB_S = g_S[N_one_third+1:2*N_one_third]
g_RB_S = g_S[2*N_one_third+1:3*N_one_third]


g_LT_N = g_N[1:N_one_third]
g_MT_N = g_N[N_one_third+1:2*N_one_third]
g_RT_N = g_N[2*N_one_third+1:3*N_one_third]




LW = e_W'
LE = e_E'
LS = e_S'
LN = e_N'

M_LB = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + β*LS'*LS*BS_y + 1/τ*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*LN'*LN + β*BS_y'*LN'*LN  # Dirichlet boundary condition on the north side

M_LM = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*LN'*LN + β*BS_y'*LN'*LN # Dirichlet boundary condition on the north side


M_LT = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + β*LN'*LN*BS_y + 1/τ*BS_y'*LN'*LN*BS_y # Neumann condition on the north side


M_MB = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + β*LS'*LS*BS_y + 1/τ*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*LN'*LN + β*BS_y'*LN'*LN  # Dirichlet boundary condition on the north side

M_MM = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*LN'*LN + β*BS_y'*LN'*LN # Dirichlet boundary condition on the north side

M_MT =  H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + β*LN'*LN*BS_y + 1/τ*BS_y'*LN'*LN*BS_y # Neumann condition on the north side

M_RB = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + β*LS'*LS*BS_y + 1/τ*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*LN'*LN + β*BS_y'*LN'*LN  # Dirichlet boundary condition on the north side

M_RM = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*LN'*LN + β*BS_y'*LN'*LN # Dirichlet boundary condition on the north side

M_RT = H_tilde*D2
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + β*LN'*LN*BS_y + 1/τ*BS_y'*LN'*LN*BS_y # Neumann condition on the north side

M_zero = zeros(N_one_third*N_one_third,N_one_third*N_one_third)



b_LB_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW' # Operators for imposing boundary conditions
b_LB_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_LB_S = τ*LS'*LS*LS' + 1/τ*BS_y'*LS'*LS*LS'
b_LB_N = τ*LN'*LN*LN' + β*BS_y'*LN'*LN*LN'

b_LM_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW'
b_LM_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_LM_S = τ*LS'*LS*LS' + β*BS_x'*LS'*LS*LS'
b_LM_N = τ*LN'*LN*LN' + β*BS_y'*LN'*LN*LN'

b_LT_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW'
b_LT_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_LT_S = τ*LS'*LS*LS' + β*BS_y'*LS'*LS*LS'
b_LT_N = τ*LN'*LN*LN' + 1/τ*BS_y'*LN'*LN*LN'


b_MB_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW' # Operators for imposing boundary conditions
b_MB_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_MB_S = τ*LS'*LS*LS' + 1/τ*BS_y'*LS'*LS*LS'
b_MB_N = τ*LN'*LN*LN' + β*BS_y'*LN'*LN*LN'

b_MM_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW'
b_MM_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_MM_S = τ*LS'*LS*LS' + β*BS_x'*LS'*LS*LS'
b_MM_N = τ*LN'*LN*LN' + β*BS_y'*LN'*LN*LN'

b_MT_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW'
b_MT_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_MT_S = τ*LS'*LS*LS' + β*BS_y'*LS'*LS*LS'
b_MT_N = τ*LN'*LN*LN' + 1/τ*BS_y'*LN'*LN*LN'

b_RB_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW' # Operators for imposing boundary conditions
b_RB_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_RB_S = τ*LS'*LS*LS' + 1/τ*BS_y'*LS'*LS*LS'
b_RB_N = τ*LN'*LN*LN' + β*BS_y'*LN'*LN*LN'

b_RM_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW'
b_RM_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_RM_S = τ*LS'*LS*LS' + β*BS_x'*LS'*LS*LS'
b_RM_N = τ*LN'*LN*LN' + β*BS_y'*LN'*LN*LN'

b_RT_W = τ*LW'*LW*LW' + β*BS_x'*LW'*LW*LW'
b_RT_E = τ*LE'*LE*LE' + β*BS_x'*LE'*LE*LE'
b_RT_S = τ*LS'*LS*LS' + β*BS_y'*LS'*LS*LS'
b_RT_N = τ*LN'*LN*LN' + 1/τ*BS_y'*LN'*LN*LN'





LW'*g_LB_W

LW'*LW
H_tilde
D2


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
