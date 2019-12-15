# Solving the Poisson Equation
# We are solving the following Poisson Equation in 2D
# Dirichlet Boundary Condition on West and East
# Neumann Boundary Condition on North and South
# Δu = -2π^2*u
# Manufactured Solution would be u(x,y) = sin(πx + πy)
# u(0,y) = sin(y)                   // Dirichlet Boundary Condition on West Side
# u(1,y) = sin(π+y) = -sin(y)       // Dirichlet Boundary Condition on East Side
# ∂u(x,0)/∂y = π*cos(π*x)   // Need negative sign    // Neumann Boundary Condition on South Side
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
i = j = 2

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

    Ax = BSx - H1x*D2x;
    Ay = BSy - H1y*D2y;




    # Forming 2d Operators
    e_1x = sparse(e(1,N_x_one_third));
    e_Nx = sparse(e(N_x_one_third,N_y_one_third));
    e_1y = sparse(e(1,N_y_one_third));
    e_Ny = sparse(e(N_y_one_third,N_y_one_third));


    I_Nx = sparse(eyes(N_x_one_third));
    I_Ny = sparse(eyes(N_y_one_third));


    # e_E = kron(e_Nx,I_Ny);
    # e_W = kron(e_1x,I_Ny);
    # e_S = kron(I_Nx,e_1y);
    # e_N = kron(I_Nx,e_Ny);
    LW = kron(I_Nx,e_1x')
    LE = kron(I_Nx,e_Nx')
    LS = kron(e_1y',I_Ny)
    LN = kron(e_Ny',I_Ny)

    # E_E = kron(sparse(Diag(e_Nx)),I_Ny);   # E_E = e_E * e_E'
    # E_W = kron(sparse(Diag(e_1x)),I_Ny);
    # E_S = kron(I_Nx,sparse(Diag(e_1y)));
    # E_N = sparse(kron(I_Nx,sparse(Diag(e_Ny))));


    D1_x = kron(D1x,I_Ny);
    D1_y = kron(I_Nx,D1y);


    D2_x = kron(D2x,I_Ny);
    D2_y = kron(I_Nx,D2y);
    D2 = D2_x + D2_y




    HI_x = kron(HIx,I_Ny);
    HI_y = kron(I_Nx,HIy);

    H_x = kron(H1x,I_Ny);
    H_y = kron(I_Nx,H1y);

    A2_x = H_x*(kron(Ax,I_Ny));
    A2_y = H_y*(kron(I_Nx,Ay));

    BS_x = kron(BSx,I_Ny);
    BS_y = kron(I_Nx,BSy);


    HI_tilde = kron(HIx,HIx);
    H_tilde = kron(H1x,H1y);


    return (D1_x, D1_y, D2_x, D2_y, Ax, Ay, A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y ,S0x, SNx, BS_x, S0y, SNy, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_1x, e_Nx, e_1y, e_Ny , LW, LE, LS, LN)
end



(D1_x, D1_y, D2_x, D2_y, Ax, Ay, A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y, S0x, SNx, BS_x, S0y, SNy, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_1x, e_Nx, e_1y, e_Ny, LW, LE, LS, LN) = Operators_2d(i,j)


## Formulate local problem
# Starting from LB block
d0 = S0x
dN = SNx

GW = -kron(H1x,d0') - kron(BS_x,e_1x')


e_1x'

d0'
