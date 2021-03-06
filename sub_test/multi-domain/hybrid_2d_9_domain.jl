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


# Interfaces: 12
# 1: LB_LM
# 2: LM_LT
#
# 3: LB_MB
# 4: LM_MM
# 5: LT_MT
#
# 6: MB_MM
# 7: MM_MT
#
# 8: MB_RB
# 9: MM_RM
# 10: MT_RT
#
# 11: RB_RM
# 12: RM_RT


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

function is_symmetric(A)
    if norm(A-A')==0
        return true
    else
        return false
    end
end

function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end



n_list = Array{Int64,1}(undef,7)
for i in range(1,step=1,stop=7)
    n_list[i] = Integer(3)^(i)
end

m_list = n_list

h_list = 1 ./ n_list

# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
# n_list = Int(1 ./h_list)

p = 2
i = j = 1

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
    # H_tilde = kron(H1x,H1y);
    Hx_tilde = kron(H1x,I_Ny);
    Hy_tilde = kron(I_Nx,H1y)
    H_tilde = Hx_tilde + Hy_tilde;

    return (D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, H_x, H_y , BS_x, BS_y, HI_tilde, Hx_tilde, Hy_tilde , H_tilde , I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N)
end

# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
# n_list = Int(1 ./h_list)

# i = j = 2
# p = 2
#
# h = h_list[i]
# n = Integer(n_list[i])
# N = n + 1
# n_one_third = Integer(n)
# N_one_third = n_one_third + 1

(D1_x, D1_y, D2_x, D2_y, D2, H_x, H_y, HI_x, HI_y, BS_x, BS_y, HI_tilde, Hx_tilde, Hy_tilde, H_tilde , I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j)

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


# e0 = e(1,N_one_third);
# en = e(N_one_third,N_one_third);
# BS = SN - S0

α = Dict(2=>1, 4=>0.2508560249, 6=>0.1878715026)
γ = 2 # γ > 1 for stability
#σ₁ = -γ/(α[p]*h_list[i]) # For left boundary Dirichlet condition
#σ₂ = -γ/(α[p]*h_list[i]) # For right boundary Neumann condition
σ₁ = -40
σ₂ = 1
β = 1
ϵ = 1  # Intersection

# g_L = 0
# g_R = -π



# NOW DEFINING HYBRID METHOD parameters

# L M R for Left Middle Right
# B M T for Bottom Middle Top
# F_LB defines external source function for block LB

F_LB = -π^2*analy_sol(span_1,span_1')
F_LM = -π^2*analy_sol(span_1,span_2')
F_LT = -π^2*analy_sol(span_1,span_3')


F_MB = -π^2*analy_sol(span_2,span_1')
F_MM = -π^2*analy_sol(span_2,span_2')
F_MT = -π^2*analy_sol(span_2,span_3')

F_RB = -π^2*analy_sol(span_3,span_1')
F_RM = -π^2*analy_sol(span_3,span_2')
F_RT = -π^2*analy_sol(span_3,span_3')

# ## Reference Definition in 1D case
# L1 = e0'
# L2 = en'
# τ = σ₁
# δ_f = 0.1
#Dδ_f = 0

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

τ = 1



M_LB =  H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + β*LS'*LS*BS_y + 1/τ*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*LN'*LN + β*BS_y'*LN'*LN  # Dirichlet boundary condition on the north side

M_LM = H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*LN'*LN + β*BS_y'*LN'*LN # Dirichlet boundary condition on the north side


M_LT = H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + β*LN'*LN*BS_y + 1/τ*BS_y'*LN'*LN*BS_y # Neumann condition on the north side


M_MB = H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + β*LS'*LS*BS_y + 1/τ*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*LN'*LN + β*BS_y'*LN'*LN  # Dirichlet boundary condition on the north side

M_MM = H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*LN'*LN + β*BS_y'*LN'*LN # Dirichlet boundary condition on the north side

M_MT =  H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + β*LN'*LN*BS_y + 1/τ*BS_y'*LN'*LN*BS_y # Neumann condition on the north side

M_RB = H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + β*LS'*LS*BS_y + 1/τ*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*LN'*LN + β*BS_y'*LN'*LN  # Dirichlet boundary condition on the north side

M_RM = H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*LN'*LN + β*BS_y'*LN'*LN # Dirichlet boundary condition on the north side

M_RT = H_x*D2_x + H_y*D2_y
        + τ*LW'*LW + β*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*LE'*LE + β*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*LS'*LS + β*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + β*LN'*LN*BS_y + 1/τ*BS_y'*LN'*LN*BS_y # Neumann condition on the north side

M_zero = zeros(N_one_third*N_one_third,N_one_third*N_one_third)


# b_LB_W here defines the opeartor to be multiplied
# with boundary condions or interface conditions


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


# Order for stacking M_matrix
# LB -> LM -> LR -> MB -> MM -> MT -> RB -> RM -> RT

function n_vcat(n::Int64,M)
    init_M = M;
    for i in 1:n-1
        init_M = vcat(init_M,M)
    end
    return init_M
end

function n_hcat(n::Int64,M)
    init_M = M;
    for i in 1:n-1
        init_M = hcat(init_M,M)
    end
    return init_M
end


M = vcat(
 hcat(M_LB,n_hcat(8,M_zero)),
 hcat(n_hcat(1,M_zero),M_LM,n_hcat(7,M_zero)),
 hcat(n_hcat(2,M_zero),M_LT, n_hcat(6,M_zero)),
 hcat(n_hcat(3,M_zero),M_MB, n_hcat(5,M_zero)),
 hcat(n_hcat(4,M_zero),M_MM, n_hcat(4,M_zero)),
 hcat(n_hcat(5,M_zero),M_MT, n_hcat(3,M_zero)),
 hcat(n_hcat(6,M_zero),M_RB, n_hcat(2,M_zero)),
 hcat(n_hcat(7,M_zero),M_RM, n_hcat(1,M_zero)),
 hcat(n_hcat(8,M_zero),M_RT))


# getting g vectors, recall g defines boundary conditions
# each term refers to one component of g vector
# We form the g vector in the same order
# LB -> LM -> LR -> MB -> MM -> MT -> RB -> RM -> RT
# The first component is g_LB, refering to the boundary conditions for
# block LB (Left-Bottom)

b_zero = zeros(N_one_third*N_one_third)

# g_LB = b_LB_W*g_LB_W + b_LB_S*g_LB_S
# g_LM = b_LM_W*g_LM_W
# g_LT = b_LT_W*g_LT_W
# g_MB = b_LB_S*g_LB_S
# g_MM = zeros(N_one_third*N_one_third)
# g_MT = b_MT_N*g_MT_N
# g_RB = b_RB_E*g_RB_E + b_RB_S*g_RB_S
# g_RM = b_RM_E*g_RM_E
# g_RT = b_RT_E*g_RT_E + b_RT_N*g_RT_N

# g = vcat(g_LB,g_LM,g_LT,g_MB,g_MM,g_MT,g_RB,g_RM,g_RT)

# We form F_T in the same order
# LB -> LM -> LR -> MB -> MM -> MT -> RB -> RM -> RT
# So the first row will be the interfaces of LB with the rest of blocks
# The first component will be the interface between LB and LM

F_zero = zeros(N_one_third*N_one_third,N_one_third)
F_T_zero = zeros(N_one_third,N_one_third*N_one_third)

# eg: F_T_LB_LM defines the component of LB-LM interface of F_T,
# refering to the term involving block LB and MB

# Constructing Interface 1: LB_LM

F_T_LB_LM_LB = τ*LN + LN*BS_y -τ*H1*LN # + τ*LE + LE*BS_x
F_T_LB_LM_LM = τ*LS + LS*BS_y - τ*H1*LS
F_T_LB_LM_LT = F_T_zero
F_T_LB_LM_MB = F_T_zero #τ*LW + LW*BS_x
F_T_LB_LM_MM = F_T_zero
F_T_LB_LM_MT = F_T_zero
F_T_LB_LM_RB = F_T_zero
F_T_LB_LM_RM = F_T_zero
F_T_LB_LM_RT = F_T_zero


F_T_LB_LM = hcat(F_T_LB_LM_LB,F_T_LB_LM_LM,F_T_LB_LM_LT,
        F_T_LB_LM_MB,F_T_LB_LM_MM,F_T_LB_LM_MT,
        F_T_LB_LM_RB,F_T_LB_LM_RM,F_T_LB_LM_LT)


# Constructing Interface 2: LM_LT

F_T_LM_LT_LB = F_T_zero # LM_LT interface does not involve LB block
F_T_LM_LT_LM = τ*LN + LN*BS_y - τ*H1*LN
F_T_LM_LT_LT = τ*LS + LS*BS_y - τ*H1*LS
F_T_LM_LT_MB = F_T_zero
# ... Trivial Terms
F_T_LM_LT = hcat(F_T_zero,F_T_LM_LT_LM,F_T_LM_LT_LT,n_hcat(6,F_T_zero))



# Constructing Interface 3: LB_MB
F_T_LB_MB_LB = τ*LE + LE*BS_x -τ*H1*LE
F_T_LB_MB_LM = F_T_zero
F_T_LB_MB_LT = F_T_zero
F_T_LB_MB_MB = τ*LW + LW*BS_x - τ*H1*LW
F_T_LB_MB_MB = F_T_zero
# ... Trivial Terms
F_T_LB_MB = hcat(F_T_LB_MB_LB,n_hcat(2,F_T_zero),F_T_LB_MB_MB,n_hcat(5,F_T_zero))

# Constructing Interface 4: LM_MM
F_T_LM_MM_LM = τ*LE + LE*BS_x - τ*H1*LE
F_T_LM_MM_MM = τ*LW + LW*BS_x - τ*H1*LW
F_T_LM_MM = hcat(F_T_zero,F_T_LM_MM_LM,n_hcat(2,F_T_zero),F_T_LM_MM_MM,n_hcat(4,F_T_zero))


# Constructing Interface 5: LT_MT
F_T_LT_MT_LT = τ*LE + LE*BS_x - τ*H1*LE
F_T_LT_MT_MT = τ*LW + LW*BS_x - τ*H1*LW
F_T_LT_MT = hcat(n_hcat(2,F_T_zero),F_T_LT_MT_LT, n_hcat(2,F_T_zero),F_T_LT_MT_MT,n_hcat(3,F_T_zero))

# Constructing Interface 6: MB_MM
F_T_MB_MM_MB = τ*LN + LN*BS_y - τ*H1*LN
F_T_MB_MM_MM = τ*LS + LS*BS_y - τ*H1*LS
F_T_MB_MM = hcat(n_hcat(3,F_T_zero),F_T_MB_MM_MB,F_T_MB_MM_MM,n_hcat(4,F_T_zero))

# Constructing Interface 7: MM_MT
F_T_MM_MT_MM = τ*LN + LN*BS_y - τ*H1*LN
F_T_MM_MT_MT = τ*LS + LS*BS_y - τ*H1*LS
F_T_MM_MT = hcat(n_hcat(4,F_T_zero),F_T_MM_MT_MM,F_T_MM_MT_MT,n_hcat(3,F_T_zero))

#
# Constructing Interface 8: MB_RB
F_T_MB_RB_MB = τ*LE + LE*BS_x - τ*H1*LE
F_T_MB_RB_RB = τ*LW + LW*BS_x - τ*H1*LW
F_T_MB_RB = hcat(n_hcat(3,F_T_zero),F_T_MB_RB_MB,n_hcat(2,F_T_zero),F_T_MB_RB_RB,n_hcat(2,F_T_zero))

# Constructing Interface 9: MM_RM
F_T_MM_RM_MM = τ*LE + LE*BS_x - τ*H1*LE
F_T_MM_RM_RM = τ*LW + LW*BS_x - τ*H1*LW
F_T_MM_RM = hcat(n_hcat(4,F_T_zero),F_T_MM_RM_MM,n_hcat(2,F_T_zero),F_T_MM_RM_RM,n_hcat(1,F_T_zero))


# Constructing Interface 10: MT_RT
F_T_MT_RT_MT = τ*LE + LE*BS_x - τ*H1*LE
F_T_MT_RT_RT = τ*LW + LW*BS_x - τ*H1*LW
F_T_MT_RT = hcat(n_hcat(5,F_T_zero),F_T_MT_RT_MT,n_hcat(2,F_T_zero),F_T_MT_RT_RT)
#


# Constructing Interface 11: RB_RM
F_T_RB_RM_RB = τ*LN + LN*BS_y - τ*H1*LN
F_T_RB_RM_RM = τ*LS + LS*BS_y - τ*H1*LS
F_T_RB_RM = hcat(n_hcat(6,F_T_zero),F_T_RB_RM_RB,F_T_RB_RM_RM,n_hcat(1,F_T_zero))

#
# Constructing Interface 12: RM_RT
F_T_RM_RT_RM = τ*LN + LN*BS_y - τ*H1*LN
F_T_RM_RT_RT = τ*LS + LS*BS_y - τ*H1*LS
F_T_RM_RT = hcat(n_hcat(7,F_T_zero),F_T_RM_RT_RM,F_T_RM_RT_RT)


# Construting Final Matrix F_T, vertical catenation of all 12 interfaces
F_T = vcat(F_T_LB_LM,F_T_LM_LT,         # First 2 interfaces
    F_T_LB_MB, F_T_LM_MM, F_T_LT_MT,    # Next 3 interfaces
    F_T_MB_MM, F_T_MM_MT,               # Next 2 interfaces
    F_T_MB_RB, F_T_MM_RM, F_T_MT_RT,    # Next 3 interfaces
    F_T_RB_RM, F_T_RM_RT)                # Next 2 interfaces


# For simplification We construct F by taking the inverse of F_T

F = F_T'


# Formulation D terms
# D represents 12 interfaces, with each interface being a 28*28 matrix
# coming from the combination of LW LE LS LN and τ
D_zero = zeros(N_one_third,N_one_third)
D_ones = ones(N_one_third, N_one_third)

N_one_third

# 1: LB_LM
D_LB_LM = D_ones*τ

# 2: LM_LT
D_LM_LT = D_ones
#
# 3: LB_MB
# 4: LM_MM
# 5: LT_MT
#
# 6: MB_MM
# 7: MM_MT
#
# 8: MB_RB
# 9: MM_RM
# 10: MT_RT
#
# 11: RB_RM
# 12: RM_RT
(D1, HI, H1, r1) = diagonal_sbp_D1(p,n_one_third,xc=(0,1/3));

# Define D matrix for each interface


D_LB_LM = H1*2τ

D = Diagonal(ones(N_one_third*12))*2τ  # Need to modify D blocks later


# Forming A terms
A = vcat(hcat(M,F),hcat(F_T,D))


# Forming g terms, g terms are the combination of source functions and boundary conditions
# each g component refers to each block, starting from block LB


#---------------------------- PREVIOUS IMPLEMENTATIONS -------------------------
# g_LB = (
#     (τ*LW' + BS_x'*LW')*g_LB_W           # LB_W
#     + (LS'+1/τ*BS_y'*LS')*g_LB_S         # LB_S
#     + H_tilde*F_LB[:])                            # Source Function for LB


# g_LM = (
#     (τ*LW' + BS_x'*LW')*g_LM_W           # LM_W
#     + H_tilde*F_LM[:])                            # Source function for LM

# g_LT = (
#     (τ*LW' + BS_x'*LW')*g_LT_W
#     + (LN'+1/τ*BS_y'*LN')*g_LT_N
#     + H_tilde*F_LT[:])


# g_MB = (
#     (LS'+1/τ*BS_y'*LS')*g_MB_S         # LB_S
#     + H_tilde*F_MB[:])                          # Source Function for LB

# g_MM = (
#     H_tilde*F_MM[:] )                            # Only source function


# g_MT = (
#     (LN'+1/τ*BS_y'*LN')*g_MT_N       # MT_N
#     + H_tilde*F_MT[:])

# g_RB = (
#     (τ*LE' + BS_x'*LE')*g_RB_E           # RB_E
#     + H_tilde*F_RB[:])

# g_RM = (
#     (τ*LE' + BS_x'*LE')*g_RM_E           # RM_E
#     + H_tilde*F_RM[:])
#
# g_RT = (
#     (τ*LE' + BS_x'*LE')*g_RT_E           # RT_E
#     + (LN'+1/τ*BS_y'*LN')*g_RT_N         # RT_N
#     + H_tilde*F_RT[:])
#

#----------------------- END OF PREVIOUS IMPLEMENTATIONS -------------------



#-------------------- NEW IMPLEMENTATIONS ---------------------------------


g_LB = (b_LB_W*g_LB_W + b_LB_S*g_LB_S) + H_tilde*F_LB[:]
g_LM = (b_LM_W*g_LM_W) + H_tilde*F_LM[:]
g_LT = (b_LT_W*g_LT_W + b_LT_N*g_LT_N) + H_tilde*F_LT[:]


g_MB = (b_MB_S * g_MB_S) + H_tilde*F_MB[:]
g_MM = H_tilde*F_MM[:]
g_MT = (b_MT_N*g_MT_N) + H_tilde*F_MT[:]


g_RB = (b_RB_E*g_RB_E + b_RB_S*g_RB_S) + H_tilde*F_RB[:]
g_RM = (b_RM_E*g_RM_E) + H_tilde*F_RM[:]
g_RT = (b_RT_E*g_RT_E + b_RT_N*g_RT_N) + H_tilde*F_RT[:]


g_bar = vcat(g_LB,g_LM,g_LT,g_MB,g_MM,g_MT,g_RB,g_RM,g_RT)


# Forming g_bar_delta
g_bar_delta = n_vcat(12,2*h*δ_f*(ones(N_one_third)))

#------------------------ END OF NEW IMPLEMENTATIONS ------------------------

# Forming b vector

b = vcat(g_bar,g_bar_delta)


lambda = (D - F_T*(M\F))\(g_bar_delta - F_T*(M\g_bar))

A\b

# g_LB == ( F_LB[:] + (τ*LW' + BS_x'*LW')*g_LB_W + (LS'+1/τ*BS_y'*LS')*g_LB_S)
# returns false didn't know why, only difference -4.440892098500626e-16





# Constructing Interface for L

# test dimensions
# LW'*g_LB_W
#
# LW'*LW
# H_tilde
# D2
# end test dimensions


# Mu =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + τ*L2'*L2 + β*BS'*L2'*L2
# Mv =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + τ*L2'*L2 + β*BS'*L2'*L2
# Mw =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + β*L2'*L2*BS + 1/τ*BS'*L2'*L2*BS#*BS #+G1
#
# # Mw = H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + β*L2'*L2*BS + 1/τ*BS'*L2'*L2*BS
#
# # F = vcat(τ*L2'+BS'*L2', τ*L1' + BS'*L1' + τ*L2' + BS'*L2' ,τ*L1'+ BS'*L1')
# # F_T = hcat(τ*L2+L2*BS, τ*L1 + L1*BS + τ*L2 + L2*BS  ,τ*L1 + L1*BS )
# b_zero = zeros(N_one_third)
#
# F = hcat(vcat(τ*L2'+BS'*L2',τ*L1'+ BS'*L1',b_zero),vcat(b_zero,τ*L2' + BS'*L2', τ*L1' + BS'*L1'))
# F_T = vcat(hcat(τ*L2+L2*BS, τ*L1 + L1*BS , b_zero'),hcat(b_zero', τ*L2 + L2*BS, τ*L1 + L1*BS))
#
# # D = 4*τ*h
# D = vcat(hcat(2*τ,0),hcat(0,2*τ))
# # D = vcat(hcat(τ,τ),hcat(τ,τ))
#
# g_bar = vcat(τ*L1'*g_L + BS'*L1'*g_L + H1*F_L, H1*F_M ,L2'*g_R + 1/τ*BS'*L2'*g_R + H1*F_R)
# # g_bar_delta = 2*h*δ_f
# g_bar_delta = vcat(2*h*δ_f,2*h*δ_f) # Not Sure
#
# Mzero = zeros(N_one_third,N_one_third)
#
# A1 = vcat(hcat(Mu,Mzero,Mzero),hcat(Mzero,Mv,Mzero),hcat(Mzero,Mzero,Mw))
#
# A = vcat(hcat(A1,F),hcat(F_T,D))
# b = vcat(g_bar,g_bar_delta)
#
# num_sol_2 = A\b
# num_sol_2_tranc = num_sol_2[1:end-2]
# plot(span,num_sol_2_tranc)
