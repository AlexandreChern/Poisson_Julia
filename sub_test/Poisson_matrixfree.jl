#Solve 2D Poisson: u_xx + u_yy = f(x,y), on the unit square with b.c.
# u(0,y) = g3(y), u(1,y) = g4(y), -u_y(x,0) = g1(x), u_y(x,1) = g2(x)
# Take the exact solution u(x,y) = sin(pi*x + pi*y)

#Transfers to discretized system
#(D2x + D2y + P1 + P2 + P3 + P4)u = b + f where
#P1 = alpha1*Hyinv*E1*BySy
#P2 = alphas2*Hyinv*E2*BySy
#P3 = alpha3*Hxinv*E3 + beta*Hxinv*BxSx_tran*E3
#P4 = alpha4*Hxinv*E4 + beta*Hxinv*BxSx_tran*E4

#b = alpha1*Hyinv*E1*g1 + alpha2*Hyinv*E2*g2 + alpha3*Hxinv*E3*g3 + beta*Hxinv*BxSx_tran*E3*g3 + ...
#    alpha4*Hxinv*E4*g4 + beta*Hxinv*BxSx_tran*E4*g4

#to make system PD, multiply by -(H kron H):

# Add sparse matrix arrays for comparison. Sparse matrix operators
# usually have underscore in their function names


include("deriv_ops.jl")
include("diagonal_sbp.jl")

using SparseArrays
using LinearMaps
using IterativeSolvers
using Parameters
using BenchmarkTools
using Plots


@with_kw struct variables
    h = 1/2^4
    dx = h
    dy = h
    x = 0:dx:1
    y = 0:dy:1
    Nx = length(x)
    Ny = length(y)
    alpha1 = -1
    alpha2 = -1
    alpha3 = -13/dy
    alpha4 = -13/dy
    beta = 1
end

var_test = variables(h=1/2^4)
@unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test

#function myMAT!(du::AbstractVector, u::AbstractVector,var_test::variables)
	#Chunk below should be passed as input, but for now needs to match chunk below
function myMAT!(du::AbstractVector, u::AbstractVector)
# 	h = 0.05
# 	dx = h
# 	dy = h
# 	x = 0:dx:1N
#         y = 0:dy:1
# 	Nx = length(x)
#         Ny = length(y)
# 	alpha1 = -1
#         alpha2 = -1
#         alpha3 = -13/dy
#         alpha4 = -13/dy
#         beta = 1
    #var_test = variables(h=0.01)
    #@unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test
	########################################

    du_ops = D2x(u,Nx,Ny,dx) + D2y(u,Nx,Ny,dy) #compute action of D2x + D2y
    du1 = BySy(u,Nx,Ny,dy)
    du2 = VOLtoFACE(du1,1,Nx,Ny)
    du3 = alpha1*Hyinv(du2,Nx,Ny,dy)  #compute action of P1

    du4 = BySy(u,Nx,Ny,dy)
    du5 = VOLtoFACE(du4,2,Nx,Ny)
    du6 = alpha2*Hyinv(du5,Nx,Ny,dy) #compute action of P2

    du7 = VOLtoFACE(u,3,Nx,Ny)
    du8 = BxSx_tran(du7,Nx,Ny,dx)
    du9 = beta*Hxinv(du8,Nx,Ny,dx)
    du10 = VOLtoFACE(u,3,Nx,Ny)
    du11 = alpha3*Hxinv(du10,Nx,Ny,dx) #compute action of P3

    du12 = VOLtoFACE(u,4,Nx,Ny)
    du13 = BxSx_tran(du12,Nx,Ny,dx)
    du14 = beta*Hxinv(du13,Nx,Ny,dx)
    du15 = VOLtoFACE(u,4,Nx,Ny)
    du16 = alpha4*Hxinv(du15,Nx,Ny,dx) #compute action of P4


    du0 = du_ops + du3 + du6 + du9 + du11 + du14 + du16 #Collect together

        #compute action of -Hx kron Hy:

    du17 = Hy(du0, Nx, Ny, dy)
	du[:] = -Hx(du17,Nx,Ny,dx)
end


function e(i,n)
    A = Matrix{Float64}(I,n,n)
    return A[:,i]
end

function eyes(n)
    return Matrix{Float64}(I,n,n)
end

function u_new(x,y)
           return sin.(π*x .+ π*y)
       end

function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end

k=2
i=j=k

function Operators_2d(i, j, p=2, h_list_x = ([1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8]),
			 h_list_y = ([1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8])
			 )
    hx = h_list_x[i];
    hy = h_list_y[j];

    x = range(0,step=hx,1);
    y = range(0,step=hy,1);
    m_list = 1 ./h_list_x;
    n_list = 1 ./h_list_y;

    # Matrix Size
    N_x = Integer(m_list[i]);
    N_y = Integer(n_list[j]);

    (D1x, HIx, H1x, r1x) = diagonal_sbp_D1(p,N_x,xc=(0,1));
    (D2x, S0x, SNx, HI2x, H2x, r2x) = diagonal_sbp_D2(p,N_x,xc=(0,1));


    (D1y, HIy, H1y, r1y) = diagonal_sbp_D1(p,N_y,xc=(0,1));
    (D2y, S0y, SNy, HI2y, H2y, r2y) = diagonal_sbp_D2(p,N_y,xc=(0,1));

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


h_list_x = [1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8]
h_list_y = [1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8]


k = 2
i = j = k

h_x = h_list_x[i]
h_y = h_list_y[j]

m_list = 1 ./h_list_x
n_list = 1 ./h_list_y

N_x = Integer(m_list[i])
N_y = Integer(n_list[j])



(D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j)


## Construct A and b with sparse matrix function

# Penalty Parameters
tau_E = -13/h_x
tau_W = -13/h_x
tau_N = -1
tau_S = -1

beta = 1

# Forming SAT terms

## Formulation 1
SAT_W = tau_W*HI_x*E_W + beta*HI_x*BS_x'*E_W
SAT_E = tau_E*HI_x*E_E + beta*HI_x*BS_x'*E_E
SAT_S = tau_S*HI_y*E_S*BS_y
SAT_N = tau_N*HI_y*E_N*BS_y

SAT_W_r = tau_W*HI_x*E_W*e_W + beta*HI_x*BS_x'*E_W*e_W
SAT_E_r = tau_E*HI_x*E_E*e_E + beta*HI_x*BS_x'*E_E*e_E
SAT_S_r = tau_S*HI_y*E_S*e_S
SAT_N_r = tau_N*HI_y*E_N*e_N




g_W = sin.(π*y)
g_E = -sin.(π*y)
g_S = -π*cos.(π*x)
g_N = π*cos.(π*x .+ π)






# Solving with CPU
A = D2 + SAT_W + SAT_E + SAT_S + SAT_N
# b = -2π^2*u(x,y')[:] + SAT_W_r*g_W + SAT_E_r*g_E + SAT_S_r*g_S + SAT_N_r*g_N


A = H_tilde*A;
b = H_tilde*b;



# @unpack h,dx,dy,x,y,Nx,Ny,alpha1,alpha2,alpha3,alpha4,beta = var_test

N = Nx*Ny
#g1 = -pi .* cos.(pi .* x)
g1 = -pi * cos.(pi * x)
#g2 = pi .* cos.(pi .* x .+ pi)
g2 = pi * cos.(pi * x .+ pi)
#g3 = sin.(pi .* y)
g3 = sin.(pi * y)
#g4 = sin.(pi .+ pi .* y)
g4 = sin.(pi .+ pi * y)

f = spzeros(Nx,Ny)
exactU = spzeros(Nx,Ny)

# for i = 1:Nx
# 	for j = 1:Ny
# 		f[j,i] = -pi^2 .* sin.(pi .* x[i] + pi .* y[j]) - pi^2 .* sin.(pi .* x[i] + pi .* y[j])
# 		exactU[j,i] = sin.(pi .* x[i] + pi .* y[j]) # bug for inconsistence in shape with exactU defined previuosly
# 	end
# end

for i = 1:Nx
	for j = 1:Ny
		f[i,j] = -pi^2 .* sin.(pi .* x[i] + pi .* y[j]) - pi^2 .* sin.(pi .* x[i] + pi .* y[j])
		exactU[i,j] = sin.(pi .* x[i] + pi .* y[j]) # bug for inconsistence in shape with exactU defined previuosly
	end
end


f = f[:]
exact = exactU[:]

#Construct vector b
b0 = FACEtoVOL(g1,1,Nx,Ny)
b1 = alpha1*Hyinv(b0,Nx,Ny,dy)

b2 = FACEtoVOL(g2,2,Nx,Ny)
b3 = alpha2*Hyinv(b2,Nx,Ny,dy)

b4 = FACEtoVOL(g3,3,Nx,Ny)
b5 = alpha3*Hxinv(b4,Nx,Ny,dx)
b6 = BxSx_tran(b4,Nx,Ny,dx)
b7 = beta*Hxinv(b6,Nx,Ny,dx)

b8 = FACEtoVOL(g4,4,Nx,Ny)
b9 = alpha4*Hxinv(b8,Nx,Ny,dx)
b10 = BxSx_tran(b8,Nx,Ny,dx)
b11 = beta*Hxinv(b10,Nx,Ny,dx)

bb = b1  + b3  + b5 + b7 + b9 + b11 + f

#Modify b for PD system
b12 = Hx(bb,Nx,Ny,dx)
b = -Hy(b12,Nx,Ny,dy)

D = LinearMap(myMAT!, N; ismutating=true)
u = cg(D,b,tol=1e-14)

plot(x,y,reshape(u,Nx,Ny),st=:surface)

diff = u - exact

Hydiff = Hy(diff,Nx,Ny,dy)
HxHydiff = Hx(Hydiff,Nx,Ny,dx)

err = sqrt(diff'*HxHydiff)

@show err

benchmark_result_rand = @benchmark cg!(rand(length(b)),D,b)
benchmark_result_norand = @benchmark cg(D,b)
display(benchmark_result_rand)
display(benchmark_result_norand)
