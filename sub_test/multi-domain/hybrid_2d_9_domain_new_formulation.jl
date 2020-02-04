# Solving the Poisson Equation
# We are solving the following Poisson Equation in 2D
# Dirichlet Boundary Condition on West and East
# Neumann Boundary Condition on North and South
# Δu = -2π^2*u
# Manufactured Solution would be u(x,y) = sin(πx + πy)
# u(0,y) = sin(y)                   // Dirichlet Boundary Condition on West Side
# u(1,y) = sin(π+y) = -sin(y)       // Dirichlet Boundary Condition on East Side
# ∂u(x,0)/∂y = -π*cos(π*x)   // Need negative sign    // Neumann Boundary Condition on South Side
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

# DEFINE ANALYICAL SOLUTIONS
#

COF_x = 1;
COF_y = 1;


function analy_sol(x,y) # Defines analytical_solution
    return sin.(COF_x*π*x .+ COF_y*π*y)
end

function u_xx(x,y)
	return -COF_x^2*π^2 .*sin.(COF_x*π*x .+ COF_y*π*y)
end

function u_yy(x,y)
	return -COF_y^2*π^2 .*sin.(COF_x*π*x .+ COF_y*π*y)
end

function u_x(x,y)
	return COF_x*π .*cos.(COF_x*π.*x .+ COF_y*π*y)
end

function u_y(x,y)
	return COF_y*π .*cos.(COF_x*π .*x .+ COF_y*π*y)
end




function is_symmetric(A)
    if norm(A-A')==0
        return true
    elseif norm(A-A') <= 1e-16
        println("Close enough");
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
i = j = 5

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

    A_tilde = kron(Ax,H1y) + kron(H1x,Ay);




    # Forming 2d Operators
    e_1x = sparse(e(1,N_x_one_third));
    e_Nx = sparse(e(N_x_one_third,N_y_one_third));
    e_1y = sparse(e(1,N_y_one_third));
    e_Ny = sparse(e(N_y_one_third,N_y_one_third));


    I_Nx = sparse(eyes(N_x_one_third));
    I_Ny = sparse(eyes(N_y_one_third));

    LW = sparse(kron(e_1y',I_Ny))
    LE = sparse(kron(e_Ny',I_Ny))
    LS = sparse(kron(I_Nx,e_1x'))
    LN = sparse(kron(I_Nx,e_Nx'))




    D1_x = sparse(kron(D1x,I_Ny));
    D1_y = sparse(kron(I_Nx,D1y));


    D2_x = sparse(kron(D2x,I_Ny));
    D2_y = sparse(kron(I_Nx,D2y));
    D2 = D2_x + D2_y




    HI_x = sparse(kron(HIx,I_Ny));
    HI_y = sparse(kron(I_Nx,HIy));

    H_x = sparse(kron(H1x,I_Ny));
    H_y = sparse(kron(I_Nx,H1y));

    A2_x = sparse(H_x*(kron(Ax,I_Ny)));
    A2_y = sparse(H_y*(kron(I_Nx,Ay)));

    BS_x = sparse(kron(BSx,I_Ny));
    BS_y = sparse(kron(I_Nx,BSy));


    HI_tilde = sparse(kron(HIx,HIx));
    H_tilde = sparse(kron(H1x,H1y));


    return (D1_x, D1_y, D2_x, D2_y, Ax, Ay, A_tilde, A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y , BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, LW, LE, LS, LN)
end

(D1_x, D1_y, D2_x, D2_y, Ax, Ay, A_tilde , A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, LW, LE, LS, LN) = Operators_2d(i,j)




span_1 = LinRange(0,1/3,N_one_third)
span_2 = LinRange(1/3,2/3,N_one_third)
span_3 = LinRange(2/3,1,N_one_third)
span= vcat(span_1,span_2,span_3)
analy_solution = analy_sol(span,span')
plot(span,span,analy_solution,st=:surface)


# e0 = e(1,N_one_third);
# en = e(N_one_third,N_one_third);
# BS = SN - S0

γ = Dict(2=>1, 4=>0.2508560249, 6=>0.1878715026)
α = H_x[1,1]/h
σ₁ = -40
σ₂ = 1
β = 1
ϵ = 1  # Intersection
#τ = 1 # Can be set as constant
τ = 2/(h*γ[p]) + 2/(h*α)
δ_f = 0 # set to zero for there is no jump




# NOW DEFINING HYBRID METHOD parameters

# L M R for Left Middle Right
# B M T for Bottom Middle Top
# F_LB defines external source function for block LB

# F were missing a factor of 2
F_LB = u_xx(span_1',span_1) .+ u_yy(span_1',span_1)
F_LM = u_xx(span_1',span_2) .+ u_yy(span_1',span_2)
F_LT = u_xx(span_1',span_3) .+ u_yy(span_1',span_3)


F_MB = u_xx(span_2',span_1) .+ u_yy(span_2',span_1)
F_MM = u_xx(span_2',span_2) .+ u_yy(span_2',span_2)
F_MT = u_xx(span_2',span_3) .+ u_yy(span_2',span_3)

F_RB = u_xx(span_3',span_1) .+ u_yy(span_3',span_1)
F_RM = u_xx(span_3',span_2) .+ u_yy(span_3',span_2)
F_RT = u_xx(span_3',span_3) .+ u_yy(span_3',span_3)




# Boundary Conditions

# L M R for Left Middle Right
# B M T for Bottom Middle Top

g_W = analy_sol(0,span)  # Boundary conditions on the west side
g_E = analy_sol(1,span) # Boundary conditions on the east side

g_S = -u_y(span',0) # add negative sing because of normal vectors # Boundary conditions on the south side

g_N = u_y(span',1) # Bondary conditions on the north side

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




LW
LE
LS
LN




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


M_LB = (-H_tilde*(D2_x+D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y # Numann boundary condition on the south Side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN) # Dirichlet boundary condition on the north side

# M_LB_test = -H_tilde*(D2_x + D2_y) + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE + H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN
# multi line operation is not the problem



M_LM = (- H_tilde*(D2_x + D2_y)  # H_x*D2_x + H_y*D2_y
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN) # Dirichlet boundary condition on the north side


M_LT =  (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + H_x*LN'*LN*BS_y - 1/τ*H_x*BS_y'*LN'*LN*BS_y) # Neumann condition on the north side



M_MB = (- H_tilde*(D2_x + D2_y) # Oh Shit I have an extra H_y here
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN) # Dirichlet boundary condition on the north side

M_MM = (- H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN) # Dirichlet boundary condition on the north side

M_MT =  (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + H_x*LN'*LN*BS_y - 1/τ*H_x*BS_y'*LN'*LN*BS_y) # Neumann condition on the north side

M_RB = (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + β*H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN)  # Dirichlet boundary condition on the north side

M_RM = (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN) # Dirichlet boundary condition on the north side

M_RT = (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + H_x*LN'*LN*BS_y - 1/τ*H_x*BS_y'*LN'*LN*BS_y) # Neumann condition on the north side

M_zero = zeros(N_one_third*N_one_third,N_one_third*N_one_third)

# M = vcat(
#  hcat(M_LB,n_hcat(8,M_zero)),
#  hcat(n_hcat(1,M_zero),M_LM,n_hcat(7,M_zero)),
#  hcat(n_hcat(2,M_zero),M_LT, n_hcat(6,M_zero)),
#  hcat(n_hcat(3,M_zero),M_MB, n_hcat(5,M_zero)),
#  hcat(n_hcat(4,M_zero),M_MM, n_hcat(4,M_zero)),
#  hcat(n_hcat(5,M_zero),M_MT, n_hcat(3,M_zero)),
#  hcat(n_hcat(6,M_zero),M_RB, n_hcat(2,M_zero)),
#  hcat(n_hcat(7,M_zero),M_RM, n_hcat(1,M_zero)),
#  hcat(n_hcat(8,M_zero),M_RT));

M = blockdiag(M_LB,M_LM,M_LT,M_MB,M_MM,M_MT,M_RB,M_RM,M_RT);


 # We form F_T in the same order
 # LB -> LM -> LR -> MB -> MM -> MT -> RB -> RM -> RT
 # So the first row will be the interfaces of LB with the rest of blocks
 # The first component will be the interface between LB and LM

 F_zero = sparse(zeros(N_one_third*N_one_third,N_one_third))
 F_T_zero = sparse(zeros(N_one_third,N_one_third*N_one_third))

 # eg: F_T_LB_LM defines the component of LB-LM interface of F_T,
 # refering to the term involving block LB and MB

 # Constructing Interface 1: LB_LM

 F_T_LB_LM_LB = (-τ*LN + β*LN*BS_y)*H_x # + τ*LE + LE*BS_x
 F_T_LB_LM_LM = (-τ*LS + β*LS*BS_y)*H_x
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


F_T_LB_LM_test = hcat(F_T_LB_LM_LB,F_T_LB_LM_LM,n_hcat(7,F_T_zero));


 # Constructing Interface 2: LM_LT

 F_T_LM_LT_LB = F_T_zero # LM_LT interface does not involve LB block
 F_T_LM_LT_LM = (-τ*LN + β*LN*BS_y)*H_x
 F_T_LM_LT_LT = (-τ*LS + β*LS*BS_y)*H_x
 F_T_LM_LT_MB = F_T_zero
 # ... Trivial Terms
 F_T_LM_LT = hcat(F_T_zero,F_T_LM_LT_LM,F_T_LM_LT_LT,n_hcat(6,F_T_zero));





 # Constructing Interface 3: LB_MB
 F_T_LB_MB_LB = (-τ*LE + β*LE*BS_x)*H_y
 F_T_LB_MB_LM = F_T_zero
 F_T_LB_MB_LT = F_T_zero
 F_T_LB_MB_MB = (-τ*LW + β*LW*BS_x)*H_y
 F_T_LB_MB_MT = F_T_zero
 # ... Trivial Terms
 F_T_LB_MB = hcat(F_T_LB_MB_LB,n_hcat(2,F_T_zero),F_T_LB_MB_MB,n_hcat(5,F_T_zero))



 # Constructing Interface 4: LM_MM
 F_T_LM_MM_LM = (-τ*LE + β*LE*BS_x)*H_y
 F_T_LM_MM_MM = (-τ*LW + β*LW*BS_x)*H_y
 F_T_LM_MM = hcat(F_T_zero,F_T_LM_MM_LM,n_hcat(2,F_T_zero),F_T_LM_MM_MM,n_hcat(4,F_T_zero))


 # Constructing Interface 5: LT_MT
 F_T_LT_MT_LT = (-τ*LE + β*LE*BS_x)*H_y
 F_T_LT_MT_MT = (-τ*LW + β*LW*BS_x)*H_y
 F_T_LT_MT = hcat(n_hcat(2,F_T_zero),F_T_LT_MT_LT, n_hcat(2,F_T_zero),F_T_LT_MT_MT,n_hcat(3,F_T_zero))

 # Constructing Interface 6: MB_MM
 F_T_MB_MM_MB = (-τ*LN + β*LN*BS_y)*H_x
 F_T_MB_MM_MM = (-τ*LS + β*LS*BS_y)*H_x
 F_T_MB_MM = hcat(n_hcat(3,F_T_zero),F_T_MB_MM_MB,F_T_MB_MM_MM,n_hcat(4,F_T_zero))

 # Constructing Interface 7: MM_MT
 F_T_MM_MT_MM = (-τ*LN + β*LN*BS_y)*H_x
 F_T_MM_MT_MT = (-τ*LS + β*LS*BS_y)*H_x
 F_T_MM_MT = hcat(n_hcat(4,F_T_zero),F_T_MM_MT_MM,F_T_MM_MT_MT,n_hcat(3,F_T_zero))

 #
 # Constructing Interface 8: MB_RB
 F_T_MB_RB_MB = (-τ*LE + β*LE*BS_x)*H_y
 F_T_MB_RB_RB = (-τ*LW + β*LW*BS_x)*H_y
 F_T_MB_RB = hcat(n_hcat(3,F_T_zero),F_T_MB_RB_MB,n_hcat(2,F_T_zero),F_T_MB_RB_RB,n_hcat(2,F_T_zero))

 # Constructing Interface 9: MM_RM
 F_T_MM_RM_MM = (-τ*LE + β*LE*BS_x)*H_y
 F_T_MM_RM_RM = (-τ*LW + β*LW*BS_x)*H_y
 F_T_MM_RM = hcat(n_hcat(4,F_T_zero),F_T_MM_RM_MM,n_hcat(2,F_T_zero),F_T_MM_RM_RM,n_hcat(1,F_T_zero))


 # Constructing Interface 10: MT_RT
 F_T_MT_RT_MT = (-τ*LE + β*LE*BS_x)*H_y
 F_T_MT_RT_RT = (-τ*LW + β*LW*BS_x)*H_y
 F_T_MT_RT = hcat(n_hcat(5,F_T_zero),F_T_MT_RT_MT,n_hcat(2,F_T_zero),F_T_MT_RT_RT)
 #


 # Constructing Interface 11: RB_RM
 F_T_RB_RM_RB = (-τ*LN + β*LN*BS_y)*H_x
 F_T_RB_RM_RM = (-τ*LS + β*LS*BS_y)*H_x
 F_T_RB_RM = hcat(n_hcat(6,F_T_zero),F_T_RB_RM_RB,F_T_RB_RM_RM,n_hcat(1,F_T_zero))

 #
 # Constructing Interface 12: RM_RT
 F_T_RM_RT_RM = (-τ*LN + β*LN*BS_y)*H_x
 F_T_RM_RT_RT = (-τ*LS + β*LS*BS_y)*H_x
 F_T_RM_RT = hcat(n_hcat(7,F_T_zero),F_T_RM_RT_RM,F_T_RM_RT_RT)


 # Construting Final Matrix F_T, vertical catenation of all 12 interfaces
 F_T = vcat(F_T_LB_LM,F_T_LM_LT,         # First 2 interfaces
     F_T_LB_MB, F_T_LM_MM, F_T_LT_MT,    # Next 3 interfaces
     F_T_MB_MM, F_T_MM_MT,               # Next 2 interfaces
     F_T_MB_RB, F_T_MM_RM, F_T_MT_RT,    # Next 3 interfaces
     F_T_RB_RM, F_T_RM_RT)                # Next 2 interfaces


 # For simplification We construct F by taking the inverse of F_T

 F = F_T'

 F = sparse(F)
 F_T = sparse(F_T)


# b_LB_W here defines the opeartor to be multiplied
# with boundary condions or interface conditions


b_LB_W = (τ*H_y*LW' - β*H_y*BS_x'*LW') # Operators for imposing boundary conditions
b_LB_E = (τ*H_y*LE' - β*H_y*BS_x'*LE')
b_LB_S = (β*H_x*LS' - 1/τ*H_x*BS_y'*LS')
b_LB_N = (τ*H_x*LN' - β*H_x*BS_y'*LN')

b_LM_W = (τ*H_y*LW' - β*H_y*BS_x'*LW')
b_LM_E = (τ*H_y*LE' - β*H_y*BS_x'*LE')
b_LM_S = (τ*H_x*LS' - β*H_x*BS_x'*LS')
b_LM_N = (τ*H_x*LN' - β*H_x*BS_y'*LN')

b_LT_W = (τ*H_y*LW' - β*H_y*BS_x'*LW')
b_LT_E = (τ*H_y*LE' - β*H_y*BS_x'*LE')
b_LT_S = (τ*H_x*LS' - β*H_x*BS_y'*LS')
b_LT_N = (β*H_x*LN' - 1/τ*H_x*BS_y'*LN')


b_MB_W = H_y*(τ*LW' - β*BS_x'*LW') # Operators for imposing boundary conditions
b_MB_E = H_y*(τ*LE' - β*BS_x'*LE')
b_MB_S = H_x*(β*LS' - 1/τ*BS_y'*LS')
b_MB_N = H_x*(τ*LN' - β*BS_y'*LN')

b_MM_W = H_y*(τ*LW' - β*BS_x'*LW')
b_MM_E = H_y*(τ*LE' - β*BS_x'*LE')
b_MM_S = H_x*(τ*LS' - β*BS_x'*LS')
b_MM_N = H_x*(τ*LN' - β*BS_y'*LN')

b_MT_W = H_y*(τ*LW' - β*BS_x'*LW')
b_MT_E = H_y*(τ*LE' - β*BS_x'*LE')
b_MT_S = H_x*(τ*LS' - β*BS_y'*LS')
b_MT_N = H_x*(β*LN' - 1/τ*BS_y'*LN')

b_RB_W = H_y*(τ*LW' - β*BS_x'*LW') # Operators for imposing boundary conditions
b_RB_E = H_y*(τ*LE' - β*BS_x'*LE')
b_RB_S = H_x*(β*LS' - 1/τ*BS_y'*LS')
b_RB_N = H_x*(τ*LN' - β*BS_y'*LN')

b_RM_W = H_y*(τ*LW' - β*BS_x'*LW')
b_RM_E = H_y*(τ*LE' - β*BS_x'*LE')
b_RM_S = H_x*(τ*LS' - β*BS_y'*LS')
b_RM_N = H_x*(τ*LN' - β*BS_y'*LN')

b_RT_W = H_y*(τ*LW' - β*BS_x'*LW')
b_RT_E = H_y*(τ*LE' - β*BS_x'*LE')
b_RT_S = H_x*(τ*LS' - β*BS_y'*LS')
b_RT_N = H_x*(β*LN' - 1/τ*BS_y'*LN')








# getting g vectors, recall g defines boundary conditions
# each term refers to one component of g vector
# We form the g vector in the same order
# LB -> LM -> LR -> MB -> MM -> MT -> RB -> RM -> RT
# The first component is g_LB, refering to the boundary conditions for
# block LB (Left-Bottom)

b_zero = sparse(zeros(N_one_third*N_one_third))

# Forming g terms, g terms are the combination of source functions and boundary conditions
# each g component refers to each block, starting from block LB

g_LB = (b_LB_W*g_LB_W + b_LB_S*g_LB_S) - H_tilde*F_LB[:]
g_LM = (b_LM_W*g_LM_W) - H_tilde*F_LM[:]
g_LT = (b_LT_W*g_LT_W + b_LT_N*g_LT_N) - H_tilde*F_LT[:]


g_MB = (b_MB_S * g_MB_S) - H_tilde*F_MB[:]
g_MM = - H_tilde*F_MM[:]
g_MT = (b_MT_N*g_MT_N) - H_tilde*F_MT[:]


g_RB = (b_RB_E*g_RB_E + b_RB_S*g_RB_S) - H_tilde*F_RB[:]
g_RM = (b_RM_E*g_RM_E) - H_tilde*F_RM[:]
g_RT = (b_RT_E*g_RT_E + b_RT_N*g_RT_N) - H_tilde*F_RT[:]


g_bar = vcat(g_LB,g_LM,g_LT,g_MB,g_MM,g_MT,g_RB,g_RM,g_RT)



# Forming g_bar_delta
g_bar_delta = n_vcat(12,2*h*δ_f*(ones(N_one_third)))




# Formulation D terms
# D represents 12 interfaces, with each interface being a 28*28 matrix
# coming from the combination of LW LE LS LN and τ
D_zero = zeros(N_one_third,N_one_third)


#D = Diagonal(ones(N_one_third*12))*2τ  # Need to modify D blocks later
D = blockdiag(H1y*2τ,H1y*2τ,
        H1x*2τ,H1x*2τ,H1x*2τ,
        H1y*2τ,H1y*2τ,
        H1x*2τ,H1x*2τ,H1x*2τ,
        H1y*2τ,H1y*2τ)

# Forming A terms
A = vcat(hcat(M,F),hcat(F_T,D))

b = vcat(g_bar,g_bar_delta)



# lambda = (D - F_T*(Matrix(M)\Matrix(F)))\(g_bar_delta - F_T*(Matrix(M)\g_bar)) # still out of memory
# \ will use LU!(), it does not support sparse Matrix
# one way is to use lu(), but we cannot write it in this way
# need some time to figure out
# M_LU = lu(M)

# lambda = (D - F_T*(M_LU.U\(M_LU.L\F[M_LU.p])))\(g_bar_delta - F_T*(M_LU.U\(M_LU.L\g_bar[M_LU.p])))
# lambda_1 = D - F_T*sparse((M_LU.U\(sparse(M_LU.L\F[M_LU.p,:]))))

lambda_2 = g_bar_delta - F_T*(M\g_bar)

lambda_1 = D - F_T*sparse(M\Matrix(F))

# lambda_2 = g_bar_delta - F_T*(M_LU.U\sparse(M_LU.L\g_bar[M_LU.p]))

lambda = lambda_1\lambda_2 # not correct


# lu(A::SparseMatrixCSC; check = true) -> F::UmfpackLU
#
# Compute the LU factorization of a sparse matrix A.
#
# For sparse A with real or complex element type, the return type of F is UmfpackLU{Tv, Ti}, with Tv = Float64 or ComplexF64 respectively
# and Ti is an integer type (Int32 or Int64).
#
# When check = true, an error is thrown if the decomposition fails. When check = false, responsibility for checking the decomposition's
# validity (via issuccess) lies with the user.
#
# The individual components of the factorization F can be accessed by indexing:
#
# Component Description
# ––––––––– –––––––––––––––––––––––––––––––
# L         L (lower triangular) part of LU
# U         U (upper triangular) part of LU
# p         right permutation Vector
# q         left permutation Vector
# Rs        Vector of scaling factors
# :         (L,U,p,q,Rs) components
#
# The relation between F and A is
#
# F.L*F.U == (F.Rs .* A)[F.p, F.q]



#num_sol = A\b # solving the system directly
num_sol = M\(g_bar - F*lambda)
#num_sol_tranc = num_sol[1:N^2*9]
plot(span,span,num_sol,st=:surface)

num_sol_LB = num_sol[1:N^2];
num_sol_LB = reshape(num_sol_LB,N,N);
sol_LB = analy_sol(span_1',span_1)
diff_LB = num_sol_LB .-sol_LB
plot(span_1,span_1,diff_LB,st=:surface)

num_sol_LM = num_sol[N^2+1:2*N^2];
num_sol_LM = reshape(num_sol_LM,N,N);
sol_LM = analy_sol(span_1',span_2)
diff_LM = num_sol_LM .- sol_LM
plot(span_1,span_2,diff_LM,st=:surface)

num_sol_LT = num_sol[2*N^2+1:3*N^2];
num_sol_LT = reshape(num_sol_LT,N,N);
sol_LT = analy_sol(span_1',span_3);
diff_LT = num_sol_LT .- sol_LT
plot(span_1,span_3,diff_LT,st=:surface)

num_sol_MB = num_sol[3*N^2+1:4*N^2];
num_sol_MB = reshape(num_sol_MB,N,N);
sol_MB = analy_sol(span_2',span_1);

num_sol_MM = num_sol[4*N^2+1:5*N^2];
num_sol_MM = reshape(num_sol_MM,N,N);
sol_MM = analy_sol(span_2',span_2);

num_sol_MT = num_sol[5*N^2+1:6*N^2];
num_sol_MT = reshape(num_sol_MT,N,N);
sol_MT = analy_sol(span_2',span_3);

num_sol_RB = num_sol[6*N^2+1:7*N^2];
num_sol_RB = reshape(num_sol_RB,N,N);
sol_RB = analy_sol(span_3',span_1);

num_sol_RM = num_sol[7*N^2+1:8*N^2];
num_sol_RM = reshape(num_sol_RM,N,N);
sol_RM = analy_sol(span_3',span_2);

num_sol_RT = num_sol[8*N^2+1:9*N^2];
num_sol_RT = reshape(num_sol_RT,N,N);
sol_RT = analy_sol(span_3',span_3);

num_sol_stacked = vcat(hcat(num_sol_LB',num_sol_MB',num_sol_RB'),
                    hcat(num_sol_LM',num_sol_MM',num_sol_RM'),
                    hcat(num_sol_LT',num_sol_MT',num_sol_RT'));
plot(span,span,num_sol_stacked,st=:surface)

@assert size(num_sol_stacked) == size(analy_solution)

diff = num_sol_stacked .- analy_solution

plot(span,span,diff,st=:surface)
savefig("./sub_test/multi-domain/plots/diff.png")



plot(span_1,span_1,num_sol_LB,st=:surface)
savefig("./sub_test/multi-domain/plots/num_sol_LB.png")
analy_sol_LB = analy_sol(span_1,span_1')
plot(span_1,span_1,analy_sol_LB,st=:surface)
savefig("./sub_test/multi-domain/plots/analy_sol_LB.png")

plot(span_1,span_2,num_sol_LM,st=:surface)
plot(span_1,span_2,sol_LM,st=:surface)

plot(span_1,span_1,reshape(M_LB\g_LB,N,N),st=:surface)
savefig("./sub_test/multi-domain/plots/num_sol_1_isolated.png")

exact = [sol_LB[:]; sol_LM[:]; sol_LT[:]; sol_MB[:]; sol_MM[:]; sol_MT[:]; sol_RB[:]; sol_RM[:]; sol_RT[:]]

num = [num_sol_LB[:]; num_sol_LM[:]; num_sol_LT[:]; num_sol_MB[:]; num_sol_MM[:]; num_sol_MT[:]; num_sol_RB[:]; num_sol_RM[:]; num_sol_RT[:]]
err = num - exact

I9 = sparse(eyes(9));
H9 = kron(I9,kron(H1x,H1y))

ERR = sqrt(err'*H9*err)
