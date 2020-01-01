# 2D Poisson Equation
# We are solving the following Poisson Equation in 2D
# Dirichlet Boundary Condition on West and East
# Neumann (Traction-free)  Boundary Condition on North and South
# Δu = f(x,y) on (x,y) in [0, 1] x [0, 1]
# Manufactured Solution we take to be u(x,y) = sin(πx + 2πy)
# u(0,y) = sin(2πy)                   // Dirichlet Boundary Condition on West Side
# u(1,y) = sin(π+2πy)                 // Dirichlet Boundary Condition on East Side
# -∂u(x,0)/∂y = -2π*cos(π*x)          // Traction-free  Boundary Condition on South Side
#  ∂u(x,1)/̡∂y = -2π*cos(πx + 2π)      // Traction-free  Boundary Condition on North Side

# Using external Julia file


# We split our domain into the following 9 blocks

# L B for Left  Right
# B T for Bottom  Top

# For each block
# W E S T for West East South North

# Eg: We refer the interfect between LB and RB (of LB) as LB_E


#interface numbering
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
#using Plots
using Pkg
Pkg.add("PyPlot")
using PyPlot

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
    return  sin.(π*x .+ 2π*y)
end

function u_xx(x,y)
	return -π^2 .* sin.(π*x .+ 2π*y)
end

function u_yy(x,y)
	return -4π^2 .* sin.(π*x .+ 2π*y)
end

function u_x(x,y)
	return π .* cos.(π*x .+ 2π*y)
end

function u_y(x,y)
	return 2π .* cos.(π*x .+ 2π*y)
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



#n_list = Array{Int64,1}(undef,7)
#for i in range(1,step=1,stop=7)
#    n_list[i] = Integer(3)^(i)
#end

n_list = [2^3 2^4 2^5 2^6]
m_list = n_list

h_list = 1 ./ n_list

EE = zeros(4,)
# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
# n_list = Int(1 ./h_list)

p = 4

i=1

j = i
h = h_list[i]

n = Integer(n_list[i])
n_one_half = Integer(n)
N = n + 1
N_one_half = n_one_half + 1

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

    n_x_one_half = n_x      # Integer(n_x)
    n_y_one_half = n_y      # Integer(n_y)

    N_x_one_half = n_x_one_half + 1
    N_y_one_half = n_x_one_half + 1

    (D1x, HIx, H1x, r1x) = diagonal_sbp_D1(p,n_x_one_half,xc=(0,1/2));
    (D2x, S0x, SNx, HI2x, H2x, r2x) = diagonal_sbp_D2(p,n_y_one_half,xc=(0,1/2));


    (D1y, HIy, H1y, r1y) = diagonal_sbp_D1(p,n_x_one_half,xc=(0,1/2));
    (D2y, S0y, SNy, HI2y, H2y, r2y) = diagonal_sbp_D2(p,n_y_one_half,xc=(0,1/2));

    BSx = sparse(SNx - S0x);
    BSy = sparse(SNy - S0y);

    Ax = BSx - H1x*D2x;
    Ay = BSy - H1y*D2y;

    A_tilde = kron(Ax,H1y) + kron(H1x,Ay);




    # Forming 2d Operators
    e_1x = sparse(e(1,N_x_one_half));
    e_Nx = sparse(e(N_x_one_half,N_y_one_half));
    e_1y = sparse(e(1,N_y_one_half));
    e_Ny = sparse(e(N_y_one_half,N_y_one_half));


    I_Nx = sparse(eyes(N_x_one_half));
    I_Ny = sparse(eyes(N_y_one_half));


    LS = kron(I_Ny,e_1x')
    LN = kron(I_Ny,e_Nx')
    LW = kron(e_1y',I_Nx)
    LE = kron(e_Ny',I_Nx)


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


    return (D1_x, D1_y, D2_x, D2_y, Ax, Ay, A_tilde, A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y , BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, LW, LE, LS, LN)
end

(D1_x, D1_y, D2_x, D2_y, Ax, Ay, A_tilde , A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, LW, LE, LS, LN) = Operators_2d(i,j)




span_1 = LinRange(0,1/2,N_one_half)
span_2 = LinRange(1/2,1,N_one_half)
span= vcat(span_1,span_2)
#analy_solution = analy_sol(span',span)



α = H_x[1,1]/h
γ = Dict(2=>0.363636363, 4=>.2505765857, 6=>0.1878715026)  #called beta in paper
β = 1
ϵ = 1  # Intersection
τ = 2/(h*γ[p]) + 2/(h*α) #Can be constant since we are doing constant coefficient problem here.
δ_f = 0 #should be 0 if solution doesn't have jumps



# NOW DEFINING HYBRID METHOD parameters


#F's were missing a factor of 2 - bae
F_LB = u_xx(span_1',span_1) .+ u_yy(span_1',span_1)
F_LT = u_xx(span_1',span_2) .+ u_yy(span_1',span_2)

F_RB = u_xx(span_2',span_1) .+ u_yy(span_2',span_1)
F_RT = u_xx(span_2',span_2) .+ u_yy(span_2',span_2)


# Boundary Conditions


g_W = analy_sol(0,span)  # Boundary conditions on the west side
g_E = analy_sol(1,span)  # Boundary conditions on the east side

g_S = -u_y(span',0)      # add negative sing because of normal vectors # Boundary conditions on the south side
g_N = u_y(span',1)       # Bondary conditions on the north side
g_LB_W = g_W[1:N_one_half]
g_LT_W = g_W[N_one_half+1:2*N_one_half]


g_RB_E = g_E[1:N_one_half]
g_RT_E = g_E[N_one_half+1:2*N_one_half]

g_LB_S = g_S[1:N_one_half]
g_RB_S = g_S[N_one_half+1:2*N_one_half]


g_LT_N = g_N[1:N_one_half]
g_RT_N = g_N[N_one_half+1:2*N_one_half]


# Order for stacking M_matrix

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


M_LB = (- H_tilde*(D2_x+D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y # Numann boundary condition on the south Side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN) # Dirichlet boundary condition on the north side




M_LT =  (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + H_x*LN'*LN*BS_y - 1/τ*H_x*BS_y'*LN'*LN*BS_y) # Neumann condition on the north side


M_RB = (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW  # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y # Numann boundary condition on the sout Side
        + τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN)  # Dirichlet boundary condition on the north side


M_RT = (-H_tilde*(D2_x + D2_y)
        + τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW # Dirichlet boundary condition on the west side
        + τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE  # Dirichlet boundary condition on the east side
        + τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS # Dirichlet boundary condition on the south side
        + H_x*LN'*LN*BS_y - 1/τ*H_x*BS_y'*LN'*LN*BS_y) # Neumann condition on the north side

M_zero = zeros(N_one_half*N_one_half,N_one_half*N_one_half)

M = vcat(
 hcat(M_LB,n_hcat(3,M_zero)),
 hcat(n_hcat(1,M_zero),M_LT, n_hcat(2,M_zero)),
 hcat(n_hcat(2,M_zero),M_RB, n_hcat(1,M_zero)),
 hcat(n_hcat(3,M_zero),M_RT))



 # Form F_T

 F_zero = zeros(N_one_half*N_one_half,N_one_half)
 F_T_zero = zeros(N_one_half,N_one_half*N_one_half)

 # Constructing Interface 1: LB_LT
 F_T_11 = (-τ*LN + β*LN*BS_y)*H_x
 F_T_12= (-τ*LS + β*LS*BS_y)*H_x
 F_T_1 = hcat(F_T_11, F_T_12, F_T_zero, F_T_zero)

 # Constructing Interface 2: LB_RB
 F_T_21 = (-τ*LE + β*LE*BS_x)*H_y
 F_T_23 = (-τ*LW + β*LW*BS_x)*H_y
 F_T_2 = hcat(F_T_21, F_T_zero, F_T_23, F_T_zero)

 # Constructing Interface 3: LT_RT
 F_T_32 = (-τ*LE + β*LE*BS_x)*H_y
 F_T_34 = (-τ*LW + β*LW*BS_x)*H_y
 F_T_3 = hcat(F_T_zero,F_T_32,F_T_zero,F_T_34)

 # Constructing Interface 4: RB_RT
 F_T_42 = (-τ*LN + β*LN*BS_y)*H_x
 F_T_44 = (-τ*LS + β*LS*BS_y)*H_x
 F_T_4 = hcat(F_T_zero, F_T_zero, F_T_42, F_T_44)


 # Construting Final Matrix F_T, vertical catenation of all 4 interfaces
 F_T = vcat(F_T_1, F_T_2, F_T_3, F_T_4)


 # For simplification We construct F by taking the transpose of F_T

F = F_T'


# b_LB_W here defines the opeartor to be multiplied
# with boundary condions or interface conditions
#H_x*LS'*LS*BS_y - c/τ*H_x*BS_y'*LS'*LS*BS_y #

b_LB_W = (τ*H_y*LW' - β*H_y*BS_x'*LW')#*LW*LW' # Operators for imposing boundary conditions
b_LB_E = (τ*H_y*LE' - β*H_y*BS_x'*LE')#*LE*LE'
b_LB_S = (H_x*LS' - 1/τ*H_x*BS_y'*LS')#*LS*LS'
b_LB_N = (τ*H_x*LN' - β*H_x*BS_y'*LN')#*LN*LN'


b_LT_W = (τ*H_y*LW' - β*H_y*BS_x'*LW')#*LW*LW'
b_LT_E = (τ*H_y*LE' - β*H_y*BS_x'*LE')#*LE*LE'
b_LT_S = (τ*H_x*LS' - β*H_x*BS_y'*LS')#*LS*LS'
b_LT_N = (H_x*LN' - 1/τ*H_x*BS_y'*LN')#*LN*LN'


b_RB_W = H_y*(τ*LW' - β*BS_x'*LW')#*LW*LW' # Operators for imposing boundary conditions
b_RB_E = H_y*(τ*LE' - β*BS_x'*LE')#*LE*LE'
b_RB_S = H_x*(LS' - 1/τ*BS_y'*LS')#*LS*LS'
b_RB_N = H_x*(τ*LN' - β*BS_y'*LN')#*LN*LN'


b_RT_W = H_y*(τ*LW' - β*BS_x'*LW')#*LW*LW'
b_RT_E = H_y*(τ*LE' - β*BS_x'*LE')#*LE*LE'
b_RT_S = H_x*(τ*LS' - β*BS_y'*LS')#*LS*LS'
b_RT_N = H_x*(LN' - 1/τ*BS_y'*LN')#*LN*LN'



# getting g vectors, recall g defines boundary conditions
# each term refers to one component of g vector
# We form the g vector in the same order
# LB -> LT -> RB  -> RT
# The first component is g_LB, refering to the boundary conditions for
# block LB (Left-Bottom)

b_zero = zeros(N_one_half*N_one_half)

# Forming g terms, g terms are the combination of source functions and boundary conditions
# each g component refers to each block, starting from block LB

g_LB = (b_LB_W*g_LB_W + b_LB_S*g_LB_S) - H_tilde*F_LB[:]
g_LT = (b_LT_W*g_LT_W + b_LT_N*g_LT_N) - H_tilde*F_LT[:]

g_RB = (b_RB_E*g_RB_E + b_RB_S*g_RB_S) - H_tilde*F_RB[:]
g_RT = (b_RT_E*g_RT_E + b_RT_N*g_RT_N) - H_tilde*F_RT[:]


g_bar = vcat(g_LB,g_LT,g_RB,g_RT)


# Forming g_bar_delta
g_bar_delta = n_vcat(4,2*h*δ_f*(ones(N_one_half)))



# Formulation D terms
D = blockdiag(H1y*2τ,H1x*2τ,H1x*2τ,H1y*2τ)

# Forming A terms
A = vcat(hcat(M,F),hcat(F_T,D))
b = vcat(g_bar,g_bar_delta)

lambda = (D - F_T*(Matrix(M)\Matrix(F)))\(g_bar_delta - F_T*(Matrix(M)\g_bar))
num_sol = M\(g_bar - F*lambda)


num_sol_LB = num_sol[1:N^2];
num_sol_LB = reshape(num_sol_LB,N,N);

num_sol_LT = num_sol[1*N^2+1:2*N^2];
num_sol_LT = reshape(num_sol_LT,N,N);

num_sol_RB = num_sol[2*N^2+1:3*N^2];
num_sol_RB = reshape(num_sol_RB,N,N);

num_sol_RT = num_sol[3*N^2+1:4*N^2];
num_sol_RT = reshape(num_sol_RT,N,N);

num_sol_stacked = vcat(hcat(num_sol_LB',num_sol_RB'),
                    hcat(num_sol_LT',num_sol_RT'));

#surf(span,span,num_sol_stacked)
#xlabel("x")
#ylabel("y")


sol_LB = analy_sol(span_1',span_1)
sol_LT = analy_sol(span_1',span_2)
sol_RB = analy_sol(span_2',span_1)
sol_RT = analy_sol(span_2',span_2)

exact = [sol_LB[:]; sol_LT[:]; sol_RB[:]; sol_RT[:]]

num = [num_sol_LB[:]; num_sol_LT[:]; num_sol_RB[:]; num_sol_RT[:]]
err = num - exact

I4 = sparse(eyes(4));
H4 = kron(I4,kron(H1x,H1y))

ERR = sqrt(err'*H4*err)

@show ERR

EE[i] = ERR

end


@show [log2(EE[1]/EE[2]) log2(EE[2]/EE[3]) log2(EE[3]/EE[4])]
