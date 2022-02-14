include("diagonal_sbp.jl")
include("deriv_ops_new.jl")
include("split_matrix_free_dev.jl")
include("split_matrix_free.jl")

# using CUDAdrv
# CUDAdrv.CuDevice(0)

# Solving Poisson Equation
# Δ u(x,y) = f(x,y)
# Manufactured Solution: u(x,y) = sin(πx .+ πy) 

# using CuArrays, CUDAnative
using LinearAlgebra
using SparseArrays
using Plots
using CUDA
using IterativeSolvers
using BenchmarkTools
using MAT

function e(i,n)
    # A = Matrix{Float64}(I,n,n)
    # return A[:,i]
    out = spzeros(n)
    out[i] = 1.0
    return out 
end

function eyes(n)
    # return Matrix{Float64}(I,n,n)
    out = spzeros(n,n)
    for i in 1:n
        out[i,i] = 1.0
    end
    return out
end

function u(x,y)
    return sin.(π*x .+ π*y)
end

function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end

function Operators_2d(i, j, p=2, h_list_x = ([1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13]),
			 h_list_y = ([1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13])
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

    # BSx = sparse(SNx - S0x);
    # BSy = sparse(SNy - S0y);
    BSx = SNx - S0x
    BSy = SNy - S0y

    # Forming 2d Operators
    # e_1x = sparse(e(1,N_x+1));
    # e_Nx = sparse(e(N_x+1,N_x+1));
    # e_1y = sparse(e(1,N_y+1));
    # e_Ny = sparse(e(N_y+1,N_y+1));
    e_1x = e(1,N_x+1);
    e_Nx = e(N_x+1,N_x+1);
    e_1y = e(1,N_x+1);
    e_Ny = e(N_y+1,N_y+1);

    # I_Nx = sparse(eyes(N_x+1));
    # I_Ny = sparse(eyes(N_y+1));
    I_Nx = eyes(N_x+1);
    I_Ny = eyes(N_y+1);


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

h_list_x = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13]
h_list_y = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13]

rel_errs = []
iter_errs = []
# for k in 1:length(h_list_x)
println("################### BEGIN TEST #########################")
for k in 6:13
    
    println()
    i = j  = k
    println("##########   Starting Test for k = ", k, "   ######################")
   
    hx = h_list_x[i];
    hy = h_list_y[j];

    x = range(0,step=hx,1);
    y = range(0,step=hy,1);
    m_list = 1 ./h_list_x;
    n_list = 1 ./h_list_y;

    # Matrix Size
    N_x = Integer(m_list[i]);
    N_y = Integer(n_list[j]);

    Nx = N_x + 1;
    Ny = N_y + 1;

    println("Nx = $Nx, Ny=$Ny")

    # 2D operators
    (D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j);


     # Analytical Solutions
    analy_sol = u(x,y');

    # Penalty Parameters
    tau_E = -13/hx;
    tau_W = -13/hx;
    tau_N = -1;
    tau_S = -1;

    beta = 1;

    # Forming SAT terms

    ## Formulation 1
    SAT_W = tau_W*HI_x*E_W + beta*HI_x*BS_x'*E_W;
    SAT_E = tau_E*HI_x*E_E + beta*HI_x*BS_x'*E_E;
    
    # SAT_S = tau_S*HI_y*E_S*D1_y
    # SAT_N = tau_N*HI_y*E_N*D1_y

    SAT_S = tau_S*HI_y*E_S*BS_y;
    SAT_N = tau_N*HI_y*E_N*BS_y;

    SAT_W_r = tau_W*HI_x*E_W*e_W + beta*HI_x*BS_x'*E_W*e_W;
    SAT_E_r = tau_E*HI_x*E_E*e_E + beta*HI_x*BS_x'*E_E*e_E;
    SAT_S_r = tau_S*HI_y*E_S*e_S;
    SAT_N_r = tau_N*HI_y*E_N*e_N;


    (alpha1,alpha2,alpha3,alpha4,beta) = (tau_N,tau_S,tau_W,tau_E,beta);


    g_W = sin.(π*y);
    g_E = -sin.(π*y);
    g_S = -π*cos.(π*x);
    # g_N = -π*cos.(π*x)
    g_N = π*cos.(π*x .+ π);

    # Solving with CPU
    A = D2 + SAT_W + SAT_E + SAT_S + SAT_N;

    b = -2π^2*u(x,y')[:] + SAT_W_r*g_W + SAT_E_r*g_E + SAT_S_r*g_S + SAT_N_r*g_N;

    A = H_tilde*A;
    b = H_tilde*b;

    @show nnz(A) * sizeof(Float64)

    @show Base.summarysize(A)
    @show Base.summarysize(b)
    @show Nx * Ny * sizeof(Float64)

    @show Base.summarysize(A) / Base.summarysize(b)
    
    println()

    A_d = CUDA.CUSPARSE.CuSparseMatrixCSC(A);
    b_d = CuArray(b);
    
    # @show Base.summarysize(A_d)
    # @show Base.summarysize(A_d)

    # testing matrix_split method
    b_reshaped = reshape(b,Nx,Ny);
    b_reshaped_GPU = CuArray(b_reshaped);

    x_GPU = similar(b_reshaped_GPU)
    x_full_GPU = similar(b_reshaped_GPU)
    A_d * b_d
    matrix_free_A(b_reshaped_GPU,x_GPU)
    matrix_free_A_full_GPU(b_reshaped_GPU,x_full_GPU)

    odata = similar(b_reshaped_GPU);
    r_GPU = similar(b_reshaped_GPU);
    p_GPU = similar(b_reshaped_GPU);
    Ap_GPU = similar(b_reshaped_GPU);

    CG_GPU_v2(b_reshaped_GPU,x_GPU,odata,r_GPU,p_GPU,Ap_GPU)



    ## SpMV test
   
    println("TESTING SpMV vs Matrix_FREE")
    iter_times = Nx + Ny

    time_matrix_free = @elapsed begin
        for _ in 1:iter_times
            matrix_free_A(b_reshaped_GPU,x_GPU)
        end
    end
    time_matrix_free = time_matrix_free * 1000 / iter_times
    @show time_matrix_free
    
    time_matrix_free_full_GPU = @elapsed begin
        for _ in 1:iter_times
            matrix_free_A_full_GPU(b_reshaped_GPU,x_full_GPU)
        end
    end
    time_matrix_free_full_GPU = time_matrix_free_full_GPU * 1000 / iter_times
    @show time_matrix_free_full_GPU
  

    time_CUBLAS = @elapsed begin
        for _ in 1:iter_times
            A_d * b_d
        end
    end
    time_CUBLAS = time_CUBLAS * 1000 / iter_times
    @show time_CUBLAS
    println()


    ## End SpMV test
    
    x_GPU = CuArray(zeros(Nx,Ny));
    (iter_steps,rsold_GPU) = CG_GPU(b_reshaped_GPU,x_GPU);
    println("Iteration steps till convergence for x_GPU: $iter_steps")
    @show rsold_GPU
    matrix_free_GPU_time_in_CG = (time_matrix_free * iter_steps)
    

    x_full_GPU = CuArray(zeros(Nx,Ny));
    (iter_steps,rsold_full_GPU) = CG_full_GPU(b_reshaped_GPU,x_full_GPU);
    matrix_free_full_GPU_time_in_CG = (time_matrix_free_full_GPU * iter_steps)
    println("Iteration steps till convergence for x_full_GPU: $iter_steps")
    @show rsold_full_GPU
   

    _,history= cg(A_d,b_d,log=true)
    @show history
    @show history.data[:resnorm][end]
    @show history.data[:resnorm][end-1]
   

    # x_GPU = CuArray(zeros(Nx,Ny));
    #         # CG_GPU_dev(b_reshaped_GPU,x_GPU)
    # CG_GPU(b_reshaped_GPU,x_GPU)

    iter_times = min(5,max(13-k,1))
    println()
    println("Starting Timing, results in ms")

   
    println("Time for matrix_free SpMV in CG: $matrix_free_GPU_time_in_CG")
    println("Time for matrix_free_full_GPU SpMV in CG: $matrix_free_full_GPU_time_in_CG")
    CUBLAS_SpMV_time_in_CG = (time_CUBLAS * history.iters)
    println("Time for CUBLAS SpMV in CG_IterativeSolvers: $CUBLAS_SpMV_time_in_CG\n")


    println()

    t_CG_GPU = @elapsed begin
        for i in 1:iter_times
            x_GPU = CuArray(zeros(Nx,Ny))
            # CG_GPU_dev(b_reshaped_GPU,x_GPU)
            CG_GPU(b_reshaped_GPU,x_GPU)
        end
    end
    t_CG_GPU = t_CG_GPU * 1000 / iter_times 
    @show t_CG_GPU

    # t_CG_GPU_v2 = @elapsed begin # testing preallocation
    #     for i in 1:iter_times
    #         x_GPU = CuArray(zeros(Nx,Ny))
    #         # CG_GPU_dev(b_reshaped_GPU,x_GPU)
    #         CG_GPU_v2(b_reshaped_GPU,x_GPU,odata,r_GPU,p_GPU,Ap_GPU)
    #     end
    # end
    # t_CG_GPU_v2 = t_CG_GPU_v2 * 1000 / iter_times 
    # @show t_CG_GPU_v2

    t_CG_full_GPU = @elapsed begin
        for i in 1:iter_times
            x_full_GPU = CuArray(zeros(Nx,Ny))
            # CG_GPU_dev(b_reshaped_GPU,x_GPU)
            CG_full_GPU(b_reshaped_GPU,x_full_GPU)
        end
    end
    t_CG_full_GPU = t_CG_full_GPU * 1000 / iter_times 
    @show t_CG_full_GPU




    t_CG_GPU_IterativeSolvers = @elapsed begin
        for i in 1:iter_times
            cg(A_d,b_d,log=true)
        end
    end
    t_CG_GPU_IterativeSolvers = t_CG_GPU_IterativeSolvers * 1000 / iter_times 
    @show t_CG_GPU_IterativeSolvers

    println("End time comparison\n")



    ## Compare Efficiency

   

    overhead_CG_matrix_free =  t_CG_GPU - matrix_free_GPU_time_in_CG 
    overhead_CG_matrix_free_full_GPU = t_CG_full_GPU - matrix_free_full_GPU_time_in_CG  
    overhead_CG_IterativeSolvrs = t_CG_GPU_IterativeSolvers - CUBLAS_SpMV_time_in_CG

    @show overhead_CG_matrix_free
    @show overhead_CG_matrix_free_full_GPU
    @show overhead_CG_IterativeSolvrs

    println()

    efficiency_CG_matrix_free = matrix_free_GPU_time_in_CG / t_CG_GPU
    efficiency_CG_matrix_free_full_GPU = matrix_free_full_GPU_time_in_CG / t_CG_full_GPU
    efficiency_CG_IterativeSolvrs = CUBLAS_SpMV_time_in_CG / t_CG_GPU_IterativeSolvers

    @show efficiency_CG_matrix_free
    @show efficiency_CG_matrix_free_full_GPU
    @show efficiency_CG_IterativeSolvrs

    println()


    ## End Compare Efficiency
    println("##########   Ending Test for k = ", k, "   ######################\n")

   
end

