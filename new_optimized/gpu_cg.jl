include("diagonal_sbp.jl")

## Loading Packages
using LinearAlgebra
using SparseArrays
using Plots
using LinearOperators

# using CuArrays, CUDAnative
using CUDA
using IterativeSolvers
using BenchmarkTools

using Dates
using DataFrames


using AlgebraicMultigrid

current_time = now()
string_time =  String((Symbol(Dates.month(current_time),'_',Dates.day(current_time),'_',Dates.hour(current_time),'_',Dates.minute(current_time))))
output_file_name = String(Symbol(string_time,".txt"))

# fileio = open("results/" * output_file_name,"w")

## Initializing Functions
function e(i,n)
    A = Matrix{Float64}(I,n,n)
    return A[:,i]
end

function se(i,n)    # Sparse formulation of e(i,n)
    A = spzeros(n)
    A[i] = 1
    return A
end

function eyes(n)
    return Matrix{Float64}(I,n,n)
end

function speyes(n)    # Sparse formulation of eyes(n)
    A = spzeros(n,n)
    for i in 1:n
        A[i,i] = 1
    end
    return A
end

function u(x,y)
           return sin.(π*x .+ π*y)
       end

function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end

# function spDiag(A)   # This is slower actually
#     n = length(A)
#     I_n = 1:n
#     J_n = 1:n
#     V_n = A
#     return sparse(I_n,J_n,V_n)
# end

function sparse_E(i,N)
    A = spzeros(N,N)
    A[i,i] = 1
    return A
end



function test_cg!(A_d,b_d)
    init_guess = similar(b_d)
    return cg!(init_guess,A_d,b_d)
end

function test_cg!_Pl(A_d,b_d)
    init_guess = similar(b_d)
    return cg!(init_guess,A_d,b_d;Pl=Identity())
end


function test_cg!_null(A_d,b_d)
    init_guess = similar(b_d)
end

function Operators_2d(i, j, hx,hy, p=2)
    # hx = h_list_x[i];
    # hy = h_list_y[j];

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

    # BSx = sparse(SNx - S0x);
    # BSy = sparse(SNy - S0y);
    BSx = SNx - S0x;
    BSy = SNy - S0y;


    # Forming 2d Operators
    # e_1x = sparse(e(1,N_x+1));
    # e_Nx = sparse(e(N_x+1,N_x+1));
    # e_1y = sparse(e(1,N_y+1));
    # e_Ny = sparse(e(N_y+1,N_y+1));
    e_1x = se(1,N_x+1)
    e_Nx = se(N_x+1,N_x+1)
    e_1y = se(1,N_y+1)
    e_Ny = se(N_y+1,N_y+1)

    #
    # I_Nx = sparse(eyes(N_x+1));
    # I_Ny = sparse(eyes(N_y+1));
    I_Nx = speyes(N_x+1)
    I_Ny = speyes(N_y+1)


    e_E = kron(e_Nx,I_Ny);
    e_W = kron(e_1x,I_Ny);
    e_S = kron(I_Nx,e_1y);
    e_N = kron(I_Nx,e_Ny);

    # E_E = kron(sparse(Diag(e_Nx)),I_Ny);   # E_E = e_E * e_E'
    # E_W = kron(sparse(Diag(e_1x)),I_Ny);
    # E_S = kron(I_Nx,sparse(Diag(e_1y)));
    # E_N = sparse(kron(I_Nx,sparse(Diag(e_Ny))));
    E_E = kron(sparse_E(N_x+1,N_x+1),I_Ny);
    E_W = kron(sparse_E(1,N_x+1),I_Ny);
    E_S = kron(I_Nx,sparse_E(1,N_y+1));
    E_N = kron(I_Nx,sparse_E(N_y+1,N_y+1));


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

h_list_x = [1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13, 1/2^14]
h_list_y = [1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13, 1/2^14]


rel_errs = []
iter_errs = []
#for k = 1:4
k = 9
println("Value for k:  ", k)
i = j  = k
hx = h_list_x[i]
hy = h_list_y[j]

x = range(0,step=hx,1)
y = range(0,step=hy,1)
m_list = 1 ./h_list_x
n_list = 1 ./h_list_y

# Matrix Size
N_x = Integer(m_list[i])
N_y = Integer(n_list[j])

# 2D operators
(D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j,hx,hy)


## Test maximum size of sparse Arrays
# p = 2
# (D1x, HIx, H1x, r1x) = diagonal_sbp_D1(p,N_x,xc=(0,1));
# (D2x, S0x, SNx, HI2x, H2x, r2x) = diagonal_sbp_D2(p,N_x,xc=(0,1));
#
#
# (D1y, HIy, H1y, r1y) = diagonal_sbp_D1(p,N_y,xc=(0,1));
# (D2y, S0y, SNy, HI2y, H2y, r2y) = diagonal_sbp_D2(p,N_y,xc=(0,1));
#
# sizeof(D1y)
#
# BSx = sparse(SNx - S0x);
# BSy = sparse(SNy - S0y);
#
# BSx = SNx - S0x;
# BSy = SNy - S0y;
#
# e_1x = se(1,N_x+1)
# e_Nx = se(N_x+1,N_x+1)
# e_1y = se(1,N_y+1)
# e_Ny = se(N_y+1,N_y+1)
#
# #
# # I_Nx = sparse(eyes(N_x+1));
# # I_Ny = sparse(eyes(N_y+1));
# I_Nx = speyes(N_x+1)
# I_Ny = speyes(N_y+1)
#
#
# e_E = kron(e_Nx,I_Ny)
# e_W = kron(e_1x,I_Ny)
# e_S = kron(I_Nx,e_1y)
# e_N = kron(I_Nx,e_Ny)
#
#
# E_E = kron(sparse_E(N_x+1,N_x+1),I_Ny)
# E_W = kron(sparse_E(1,N_x+1),I_Ny)
# E_S = kron(I_Nx,sparse_E(1,N_y+1))
# E_N = kron(I_Nx,sparse_E(N_y+1,N_y+1))
#
#
# D1_x = kron(D1x,I_Ny)
# D1_y = kron(I_Nx,D1y)
#
#
# D2_x = kron(D2x,I_Ny)
# D2_y = kron(I_Nx,D2y)
# D2 = D2_x + D2_y
#
#
# HI_x = kron(HIx,I_Ny)
# HI_y = kron(I_Nx,HIy)
#
# H_x = kron(H1x,I_Ny)
# H_y = kron(I_Nx,H1y)
#
# BS_x = kron(BSx,I_Ny)
# BS_y = kron(I_Nx,BSy)
#
#
# HI_tilde = kron(HIx,HIx)
# H_tilde = kron(H1x,H1y)
## Start formulating system

# Analytical Solutions
analy_sol = u(x,y')
cu_analy_sol = CuArray(analy_sol)

# Penalty Parameters
tau_E = -13/hx
tau_W = -13/hx
tau_N = -1
tau_S = -1

beta = 1

# Forming SAT terms

## Formulation 1
SAT_W = tau_W*HI_x*E_W + beta*HI_x*BS_x'*E_W;
SAT_E = tau_E*HI_x*E_E + beta*HI_x*BS_x'*E_E;
SAT_S = tau_S*HI_y*E_S*BS_y;
SAT_N = tau_N*HI_y*E_N*BS_y;

SAT_W_r = tau_W*HI_x*E_W*e_W + beta*HI_x*BS_x'*E_W*e_W;
SAT_E_r = tau_E*HI_x*E_E*e_E + beta*HI_x*BS_x'*E_E*e_E;
SAT_S_r = tau_S*HI_y*E_S*e_S;
SAT_N_r = tau_N*HI_y*E_N*e_N;




g_W = sin.(π*y)
g_E = -sin.(π*y)
g_S = -π*cos.(π*x)
g_N = π*cos.(π*x .+ π)






# Solving with CPU
A = D2 + SAT_W + SAT_E + SAT_S + SAT_N;
b = -2π^2*u(x,y')[:] + SAT_W_r*g_W + SAT_E_r*g_E + SAT_S_r*g_S + SAT_N_r*g_N;


A = H_tilde*A;
b = H_tilde*b;

A_fac = lu(A)


println("Time for factorization:")
test_lu = @benchmark A_fac = lu(A)
display(test_lu)

println("Time for direct solve:")
test_solve = @benchmark A_fac \ b
display(test_solve)

direct_sol = A_fac \ b
direct_err = sqrt((direct_sol[:] - analy_sol[:])' * H_tilde * (direct_sol[:] - analy_sol[:]))
@show direct_err

## Solving with GPU
A_d = CuArrays.CUSPARSE.CuSparseMatrixCSC(A)
# b_d = sparse(CuArray{Float64}(b))

b_d = CuArray(b)

# b_d = CuArrays.CUSPARSE.CuSparseMatrixCSC(b)

init_guess = rand(length(b))
init_guess_copy = init_guess;
init_guess = CuArray{Float64}(init_guess);
init_guess_v2 = CuArray{Float64}(rand(length(b)))

# Numerical Solutions

#result_1 = @benchmark A\b

#num_sol = A\b
#num_sol = reshape(num_sol, N_y+1, N_x + 1)
#num_err = sqrt((num_sol[:] - analy_sol[:])' * H_tilde * (num_sol[:] - analy_sol[:]))
#log_num_err = log2.(num_err)

## Iterative Solutions
## GPU

# result_2 = @benchmark cg!(init_guess,A_d,b_d;tol=1e-16)
#result_2 = @benchmark cg(A_d,b_d)

cu_sol = cg!(init_guess,A_d,b_d)
cu_sol = collect(cu_sol)
cu_sol = reshape(cu_sol, N_y + 1, N_x + 1)
iter_GPU_err = sqrt((cu_sol[:] - analy_sol[:])' * H_tilde * (cu_sol[:] - analy_sol[:]))


function calculate_cg_err(A_d,b_d,maxiter_num)
    cu_sol = cg(A_d,b_d;maxiter=maxiter_num)
    cu_sol = collect(cu_sol)
    iter_GPU_err = sqrt((cu_sol - analy_sol[:])' * H_tilde * (cu_sol - analy_sol[:]))
    return iter_GPU_err
end


# calculate_cg_err(A_d,b_d,100)
# calculate_cg_err(A_d,b_d,200)
# calculate_cg_err(A_d,b_d,300)
# calculate_cg_err(A_d,b_d,400)
# calculate_cg_err(A_d,b_d,500)
# calculate_cg_err(A_d,b_d,600)
# calculate_cg_err(A_d,b_d,700)
# calculate_cg_err(A_d,b_d,800)
# calculate_cg_err(A_d,b_d,900)
# calculate_cg_err(A_d,b_d,1000)
# calculate_cg_err(A_d,b_d,1100)
# calculate_cg_err(A_d,b_d,1200)
# calculate_cg_err(A_d,b_d,1300)
# calculate_cg_err(A_d,b_d,1400)
# calculate_cg_err(A_d,b_d,1500)
# calculate_cg_err(A_d,b_d,1600)
# calculate_cg_err(A_d,b_d,1700)
# calculate_cg_err(A_d,b_d,1800)
# calculate_cg_err(A_d,b_d,1900)
# calculate_cg_err(A_d,b_d,2000)


# b_d_2 = CuArray(b)
# cu_sol_v2 = cg!(init_guess_v2,A_d,b_d_2;maxiter=10000)
# cu_sol_v2 = collect(cu_sol_v2)
# cu_sol_v2 = reshape(cu_sol_v2,N_y+1, N_x+1)
# iter_GPU_err_v2 = sqrt((cu_sol_v2[:] - analy_sol[:])' * H_tilde * (cu_sol_v2[:] - analy_sol[:]))
#
# log_iter_GPU_err = log2.(iter_GPU_err)
# log_iter_GPU_err_v2 = log2.(iter_GPU_err_v2)

result_2 = @benchmark test_cg!(A_d,b_d)

@benchmark test_cg!_Pl(A_d,b_d)

# result = @benchmark test_cg!(A_d,b_d)

cu_sol - analy_sol
# cu_sol_v2 - analy_sol




function iter_err_by_steps(;div_num=100)
    iter_err_lists = []

    dim = size(A)[1]

    div(dim,div_num)
    max_steps = [div(i*dim,div_num) for i in 1:div_num]
    for step in max_steps
        @show step
        cu_sol = cg(A_d,b_d;maxiter=step)
        cu_sol = collect(cu_sol)
        cu_sol = reshape(cu_sol, N_y + 1, N_x + 1)
        iter_GPU_err = sqrt((cu_sol[:] - analy_sol[:])' * H_tilde * (cu_sol[:] - analy_sol[:]))
        push!(iter_err_lists,iter_GPU_err)
    end
    iter_err_lists
    vcat(max_steps, iter_err_lists)
    DataFrame("max_steps"=>max_steps,"iter_err"=>iter_err_lists)
    plot(max_steps,iter_err_lists)
    savefig("plots/error_$dim.png")
    plot(max_steps,log.(iter_err_lists))
    savefig("plots/log_error_$dim.png")
end

# iter_err_by_steps(div_num=100)

## CPU  using BLAS
#result_3 = @benchmark cg!(init_guess_copy,A,b)
##result_3 = @benchmark cg(A,b)
#iter_sol = cg!(init_guess_copy,A,b)
#iter_sol = cg(A,b)
#iter_sol = reshape(iter_sol,N_y+1, N_x+1)
#iter_CPU_err = sqrt((iter_sol[:] - analy_sol[:])' * H_tilde * (iter_sol[:] - analy_sol[:]))
#log_iter_CPU_err = log2.(iter_CPU_err)

#rel_err = sqrt(err)
#rel_iter_err = sqrt(iter_err)



#push!(rel_errs,rel_err)
#push!(iter_errs,iter_err)

#println("For CPU LU Decomposition:")
#display(result_1)
#println()


# println("For GPU Iterative:")
# # write(file_io,"For GPU Iterative: \n")
# display(result_2)
# println()


#println("For CPU Iterative")
#display(result_3)
#println()

#println("Error Comparisons")
#println("For CPU LU Decomposition:")
#println(num_err)
#println(log_num_err)
#println()
#
# println("For GPU Iterative:")
# println(iter_GPU_err)
# println(log_iter_GPU_err)
# println()

##println("For CPU Iterative:")
#println(iter_CPU_err)
#println(log_iter_CPU_err)
#println()
# close(file_io)
#end
