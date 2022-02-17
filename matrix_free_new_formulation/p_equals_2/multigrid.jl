include("diagonal_sbp.jl")
include("deriv_ops_new.jl")
include("split_matrix_free_dev.jl")
include("split_matrix_free.jl")


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

h_list_x = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13, 1/2^14]
h_list_y = [1/2^1, 1/2^2, 1/2^3, 1/2^4, 1/2^5, 1/2^6, 1/2^7, 1/2^8, 1/2^9, 1/2^10, 1/2^11, 1/2^12, 1/2^13, 1/2^14]

function Operators_2d(i, j, h_list_x, h_list_y; p=2)
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

function Assembling_matrix(level)
    i = j = level
    hx = h_list_x[i];
    hy = h_list_y[j];

    x = range(0,step=hx,1);
    y = range(0,step=hy,1);
    m_list = 1 ./h_list_x;
    n_list = 1 ./h_list_y;

    N_x = Integer(m_list[i]);
    N_y = Integer(n_list[j]);

    Nx = N_x + 1;
    Ny = N_y + 1;

    (D1_x, D1_y, D2_x, D2_y, D2, HI_x, HI_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, e_E, e_W, e_S, e_N, E_E, E_W, E_S, E_N) = Operators_2d(i,j,h_list_x,h_list_y);
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

    A = -H_tilde*A;
    b = -H_tilde*b;

    return (A,b,H_tilde,Nx,Ny)
end


function check_memory_allocations_for_factorization(lower_level,upper_level)
    for level in lower_level:upper_level
        (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level)
        @show Nx,Ny
        allocated_memory_lu = @allocated lu(A)
        println("Allocated memory to store A: $(Base.summarysize(A))")
        println("Allocated memory for LU factorization: $allocated_memory_lu")
        println("Ratio malloc(lu(A)) / malloc(A): $(allocated_memory_lu/Base.summarysize(A))")
        lu_A = lu(A)
        allocated_memory_lu_solving = @allocated lu_A \ b
        println("Memory allocation for solving A\b with lu factorization: $allocated_memory_lu_solving")

        allocated_memory_cholesky = @allocated cholesky(A)
        println("Allocated memory for Cholesky factorization: $allocated_memory_cholesky")
        println("Ratio malloc(cholesky(A)) / malloc(A): $(allocated_memory_cholesky/Base.summarysize(A))")
        cholesky_A = cholesky(A)
        allocated_memory_cholesky_solving = @allocated cholesky_A \ b
        println("Memory allocation for solving A\b with cholesky factorization: $allocated_memory_cholesky_solving")
        println()
    end
end


function matrix_prolongation(idata)
    (Nx,Ny) = size(idata)
    odata_Nx = 2*Nx-1
    odata_Ny = 2*Ny-1
    odata = zeros(odata_Nx,odata_Ny)
    for i in 1:odata_Nx
        for j in 1:odata_Ny
            if i % 2 == 1
                if j % 2 == 1
                    odata[i,j] = idata[div(i,2)+1,div(j,2)+1]
                else 
                    odata[i,j] = (idata[div(i,2)+1,div(j,2)] +  idata[div(i,2)+1,div(j,2)+1]) / 2
                end
            else
                if j % 2 == 1
                    odata[i,j] = (idata[div(i,2),div(j,2)+1] +  idata[div(i,2)+1,div(j,2)+1]) / 2
                else 
                    odata[i,j] = (idata[div(i,2),div(j,2)] +  idata[div(i,2)+1,div(j,2)] + idata[div(i,2),div(j,2) + 1] +  idata[div(i,2)+1,div(j,2) + 1]) / 4
                end
            end
        end
    end
    return odata
end


function matrix_restriction(idata)
    (Nx,Ny) = size(idata)
    odata_Nx = div(Nx+1,2)
    odata_Ny = div(Ny+1,2)
    odata = zeros(odata_Nx,odata_Ny)
    for i in 1:odata_Nx
        for j in 1:odata_Ny
            odata[i,j] = idata[2*i-1,2*j-1]
        end
    end
    return odata
end


function prolongation_matrix(N)
    # SBP preserving
    # N = 2^level + 1
    odata = spzeros(2*N-1,N)
    for i in 1:2*N-1
        if i % 2 == 1
            odata[i,div(i+1,2)] = 1
        else
            odata[i,div(i,2)] = 1/2
            odata[i,div(i,2)+1] = 1/2
        end
    end
    return odata
end

function restriction_matrix(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/2
    odata[end,end-1] = 1/2
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function restriction_matrix_normal(N)
    # SBP preserving
    odata = spzeros(div(N+1,2),N)
    odata[1,1] = 1/2
    odata[1,2] = 1/4
    odata[end,end-1] = 1/4
    odata[end,end] = 1/2
    for i in 2:div(N+1,2)-1
        odata[i,2*i-2] = 1/4
        odata[i,2*i-1] = 1/2
        odata[i,2*i] = 1/4
    end
    return odata
end

function prolongation_2d(N)
    prolongation_1d = prolongation_matrix(N)
    prolongation_2d = kron(prolongation_1d,prolongation_1d)
    return prolongation_2d
end

function restriction_2d(N)
    restriction_1d = restriction_matrix_normal(N)
    restriction_2d = kron(restriction_1d,restriction_1d)
    return restriction_2d
end


# function Jacobi_iter(x,A,b)
#     k = 0
#     n = length(x)
#     iter_times = 0
#     # while norm(A*x-b) >= 1e-8
#     for _ in 1:10
#         iter_times += 1
#         for i = 1:n
#             σ = 0
#             for j = 1:n
#                 if j != i
#                     σ = σ + A[i,j]*x[j]
#                 end
#             end
#             x[i] = (b[i] - σ)/A[i,i]
#         end
#     end
#     (x,iter_times)
# end

function jacobi_iter!(x,A,b;maxiter=4)
    k = 0
    n = length(x)
    iter_times = 0
    # while norm(A*x-b) >= 1e-8
    for _ in 1:maxiter
        iter_times += 1
        for i = 1:n
            σ = 0
            for j = 1:n
                if j != i
                    σ = σ + A[i,j]*x[j]
                end
            end
            x[i] = (b[i] - σ)/A[i,i]
        end
    end
    # (x,iter_times)
end

function Two_Grid_Correction(;level=9,nu=10,use_galerkin=true)
    # level = 9
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    (A_p,b_p,H_tilde_p,Nx_p,Ny_p) = Assembling_matrix(level-1);

    x = zeros(length(b));
    jacobi!(x,A,b;maxiter=nu*2);
    
    println("Without coarse grid correction, norm(A*x-b) after $(nu*2) iterations: $(norm(A*x-b))")

    x = zeros(length(b));
    x_p = zeros(length(b_p));
    e_fine = zeros(length(b));

    jacobi!(x,A,b;maxiter=nu);
    r = b - A*x;
    # @show norm(r)
    println("Without coarse grid correction, norm(A*x-b) after 10 iterations: $(norm(A*x-b))")

    f = restriction_2d(Nx) * r;
    A_coarse = restriction_2d(Nx) * A * prolongation_2d(Nx_p);
    # jacobi!(x_p,A_coarse,f);
    if use_galerkin
        x_p = A_coarse \ f
    else
        x_p = A_p \ f
    end 
    # jacobi!(x_p,A_p,f);
    e_1 = prolongation_2d(Nx_p) * x_p;
    x = x + e_1;
    println("After coarse grid correction, norm(A*x-b): $(norm(A*x-b))")
    jacobi!(x,A,b;maxiter=nu);
    println("After coarse grid correction, norm(A*x-b) after another 10 iterations: $(norm(A*x-b))")
end


A_matrices = Dict()

function multigrid(A_matrices;level=8,L=3,nu=10,NUM_V_CYCLES=1,use_galerkin=true)
    # level = 8
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    # L = 3 # multigrid level
    v_values = Dict(1=>zeros(Nx*Ny))
    rhs_values = Dict(1 => b)

    # @show A_matrices
    @show isempty(A_matrices)
    if isempty(A_matrices)
        # A_matrices = Dict(1 => A)
        A_matrices[1] = A
        # A_matrices[1] = lu(A)
        N_values = Dict(1=>Nx)
        # NUM_V_CYCLES = 2
        for cycle_number in 1:NUM_V_CYCLES
            @show cycle_number
            # max_iter = 10
            for i in 1:L
                if i != L
                    # @show i
                    @show i, norm(rhs_values[i]) / (N_values[i] - 1)
                    jacobi!(v_values[i],A_matrices[i],rhs_values[i];maxiter=nu)
                    # @show i 
                    # @show size(restriction_2d(N_values[i]))
                    # @show size(rhs_values[i]) 
                    # @show size(A_matrices[i])
                    # @show size(v_values[i])
                    @show i, norm(A_matrices[i] * v_values[i] - rhs_values[i]) * 1/ (N_values[i] - 1)
                    rhs_values[i+1] = restriction_2d(N_values[i]) * (rhs_values[i] - A_matrices[i] * v_values[i])
                    N_values[i+1] = div(N_values[i]+1,2)
                    if use_galerkin
                        A_matrices[i+1] = restriction_2d(N_values[i]) * A_matrices[i] * prolongation_2d(N_values[i+1])
                    else
                        A_matrices[i+1] = Assembling_matrix(level-i)[1]
                    end
                    v_values[i+1] = zeros(N_values[i+1]^2)
                else
                    v_values[i] = A_matrices[i] \ rhs_values[i]
                end
                
            end

            # println("Pass first part")

            for i in 1:L-1
                j = L-i
                @show j, norm(rhs_values[j]) / (N_values[j] - 1)
                v_values[j] = v_values[j] + prolongation_2d(N_values[j+1]) * v_values[j+1]
                jacobi!(v_values[j],A_matrices[j],rhs_values[j];maxiter=nu)
                @show j, norm(A_matrices[j]*v_values[j] - rhs_values[j]) * 1/ (N_values[j] - 1)
            end
            # @show norm(A_matrices[1] * v_values[1] - b)
        end
        return (v_values[1],norm(A_matrices[1] * v_values[1] - b))
    else
        N_values = Dict(1=>Nx)
        # NUM_V_CYCLES = 2
        for cycle_number in 1:NUM_V_CYCLES
            @show cycle_number
            # max_iter = 10
            for i in 1:L
                if i != L
                    # @show i
                    @show i, norm(rhs_values[i]) / (N_values[i] - 1)
                    jacobi!(v_values[i],A_matrices[i],rhs_values[i];maxiter=nu)
                    @show i, norm(A_matrices[i] * v_values[i] - rhs_values[i]) * 1/ (N_values[i] - 1)
                    # @show i 
                    # @show size(restriction_2d(N_values[i]))
                    # @show size(rhs_values[i]) 
                    # @show size(A_matrices[i])
                    # @show size(v_values[i])
                    rhs_values[i+1] = restriction_2d(N_values[i]) * (rhs_values[i] - A_matrices[i] * v_values[i])
                    N_values[i+1] = div(N_values[i]+1,2)
                    # if use_galerkin
                    #     A_matrices[i+1] = restriction_2d(N_values[i]) * A_matrices[i] * prolongation_2d(N_values[i+1])
                    # else
                    #     A_matrices[i+1] = Assembling_matrix(level-i)[1]
                    # end
                    v_values[i+1] = zeros(N_values[i+1]^2)
                else
                    v_values[i] = A_matrices[i] \ rhs_values[i]
                end
                
            end

            # println("Pass first part")

            for i in 1:L-1
                j = L-i
                @show j, norm(rhs_values[j]) / (N_values[j] - 1)
                v_values[j] = v_values[j] + prolongation_2d(N_values[j+1]) * v_values[j+1]
                jacobi!(v_values[j],A_matrices[j],rhs_values[j];maxiter=nu)
                @show j, norm(A_matrices[j] * v_values[j] - rhs_values[j]) * 1 / (N_values[j]-1)
            end
            # @show norm(A_matrices[1] * v_values[1] - b)
        end
        # return (v_values[1],norm(A_matrices[1] * v_values[1] - b) / (N_values[1] - 1))
        return norm(A_matrices[1] * v_values[1] - b) / (N_values[1] - 1)
    end
end


function mg(A,b,L=mg_level;nu=4,NUM_V_CYCLES=1,use_galerkin=true)
    Nx = Ny = Int(sqrt(length(b)))
    level = Int(log(2,Nx-1))
    v_values = Dict(1=>zeros(Nx*Ny))
    rhs_values = Dict(1 => b)
    A_matrices = Dict(1 => A)
    N_values = Dict(1=>Nx)
    # @show L
    for _ in 1:NUM_V_CYCLES
        for i in 1:L
            if i != L
                # @show i
                # jacobi!(v_values[i],A_matrices[i],rhs_values[i];maxiter=nu)
                # @show size(restriction_2d(N_values[i]))
                # @show size(rhs_values[i]) 
                # @show size(A_matrices[i])
                # @show size(v_values[i])
                rhs_values[i+1] = restriction_2d(N_values[i]) * (rhs_values[i] - A_matrices[i] * v_values[i])
                N_values[i+1] = div(N_values[i]+1,2)
                if use_galerkin
                    A_matrices[i+1] = restriction_2d(N_values[i]) * A_matrices[i] * prolongation_2d(N_values[i+1])
                else
                    A_matrices[i+1] = Assembling_matrix(level-i)[1]
                end
                v_values[i+1] = zeros(N_values[i+1]^2)
            else
                v_values[i] = A_matrices[i] \ rhs_values[i]
            end
            
        end

        # println("Pass first part")

        for i in 1:L-1
            j = L-i
            v_values[j] = v_values[j] + prolongation_2d(N_values[j+1]) * v_values[j+1]
            jacobi!(v_values[j],A_matrices[j],rhs_values[j];maxiter=nu)
        end
    end
    return v_values[1]
end

function mg_v2(A,b,A_matrices,L=mg_level;nu=4,NUM_V_CYCLES=1,use_galerkin=true)
    Nx = Ny = Int(sqrt(length(b)))
    level = Int(log(2,Nx-1))
    v_values = Dict(1=>zeros(Nx*Ny))
    rhs_values = Dict(1 => b)
    # A_matrices = Dict(1 => A)
    N_values = Dict(1=>Nx)
    
    if isempty(A_matrices)
        println("A_matrices is empty, starting to assemble")
        A_matrices[1] = A
        # @show L
        for _ in 1:NUM_V_CYCLES
            for i in 1:L
                if i != L
                    # @show i
                    jacobi!(v_values[i],A_matrices[i],rhs_values[i];maxiter=nu)
                    # ssor!(v_values[i],A_matrices[i],rhs_values[i],2/3;maxiter=nu)
                    # @show size(restriction_2d(N_values[i]))
                    # @show size(rhs_values[i]) 
                    # @show size(A_matrices[i])
                    # @show size(v_values[i])
                    rhs_values[i+1] = restriction_2d(N_values[i]) * (rhs_values[i] - A_matrices[i] * v_values[i])
                    N_values[i+1] = div(N_values[i]+1,2)
                    if use_galerkin
                        A_matrices[i+1] = restriction_2d(N_values[i]) * A_matrices[i] * prolongation_2d(N_values[i+1])
                    else
                        A_matrices[i+1] = Assembling_matrix(level-i)[1]
                    end
                    v_values[i+1] = zeros(N_values[i+1]^2)
                else
                    A_matrices[i] = lu(A_matrices[i])
                    v_values[i] = A_matrices[i] \ rhs_values[i]
                end
                
            end

            # println("Pass first part")

            for i in 1:L-1
                j = L-i
                v_values[j] = v_values[j] + prolongation_2d(N_values[j+1]) * v_values[j+1]
                jacobi!(v_values[j],A_matrices[j],rhs_values[j];maxiter=nu)
            end
        end
        return v_values[1]
    else
        # println("A_matrices is not empty, using existing results")
        for _ in 1:NUM_V_CYCLES
            for i in 1:L
                if i != L
                    # @show i
                    # jacobi!(v_values[i],A_matrices[i],rhs_values[i];maxiter=nu) # forget to do this iteration
                    # @show size(restriction_2d(N_values[i]))
                    # @show size(rhs_values[i]) 
                    # @show size(A_matrices[i])
                    # @show size(v_values[i])
                    rhs_values[i+1] = restriction_2d(N_values[i]) * (rhs_values[i] - A_matrices[i] * v_values[i])
                    N_values[i+1] = div(N_values[i]+1,2)
                    # if use_galerkin
                    #     A_matrices[i+1] = restriction_2d(N_values[i]) * A_matrices[i] * prolongation_2d(N_values[i+1])
                    # else
                    #     A_matrices[i+1] = Assembling_matrix(level-i)[1]
                    # end
                    v_values[i+1] = zeros(N_values[i+1]^2)
                else
                    v_values[i] = A_matrices[i] \ rhs_values[i]
                end
                
            end

            # println("Pass first part")

            for i in 1:L-1
                j = L-i
                v_values[j] = v_values[j] + prolongation_2d(N_values[j+1]) * v_values[j+1]
                jacobi!(v_values[j],A_matrices[j],rhs_values[j];maxiter=nu)
            end
        end
        return v_values[1]
    end
end

function pure_jacobi(;level=8,nu=10)
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    x = zeros(Nx*Ny)
    jacobi!(x,A,b,maxiter=nu)
    return norm(A*x - b)
end


function test_multigrid_CG()
    # cg!(x,A,b,abstol=1e-8,log=true)
    # cg!(x_p,A_p,b_p,abstol=1e-8,log=true)

    # x = zeros(length(b))
    # r1 = A*x - b

    # r2 = matrix_restriction(reshape(r1,Nx,Ny))[:]
    # cg!(r2,A_p,zeros(length(b_p)),abstol=1e-6,log=true)
    # r1 = matrix_prolongation(reshape(r2,Nx_p,Ny_p))[:];
    # cg!(r1,A,zeros(length(b)),abstol=1e-8,log=true)

    # cg!(r1,A,b,abstol=1e-8,log=true)
end



function richardson(x,A,b;maxiter=10)
    ω = -1/14
    for _ in 1:maxiter
        x .= x .+ ω*(b - A*x)
    end
    return norm(A*x-b)
end



function jacobi_smoothed_CG(A,b,x;maxiter=length(b),jacobi_iter=10,abstol=sqrt(eps(real(eltype(b)))))
    r = b - A * x;
    rnew = similar(r)
    z = jacobi(A,r,maxiter=jacobi_iter)
    znew = similar(z)
    p = z;
    Ap = A*p;
    num_iter_steps = 0
    for step = 1:maxiter
    # for _ in 1:40
        num_iter_steps += 1
        alpha = r'*z/(p'*Ap)
        mul!(Ap,A,p);
        alpha = r'*z / (p'*Ap)
        x .= x .+ alpha * p;
        rnew .= r .- alpha * Ap;
        rsnew = rnew' * rnew
        if sqrt(rsnew) < abstol
              break
        end
        # p = r + (rsnew / rsold) * p;
        znew = jacobi(A,rnew,maxiter=jacobi_iter=10);
        beta = rnew'*znew/(r'*z);
        p .= znew .+ beta * p;
        z .= znew
        r .= rnew
        @show step, rsnew
    end
    # @show num_iter_steps
    num_iter_steps
end


function jacobi_preconditioned_CG(A,b,x)
    r = b - A * x;
    # z = jacobi(A,r,maxiter=10)
    M = spdiagm(diag(A))
    z = M \ r
    # z = A \ r
    p = z;
    Ap = A*p;
    alpha = r'*z/(p'*Ap);
    x .= x .+ alpha * p;
    r .= r .- alpha * Ap;
    num_iter_steps = 0
    for _ = 1:length(b)
    # for _ in 1:40
        num_iter_steps += 1
        # znew = jacobi(A,r,maxiter=10)
        znew = M \ r 
        # znew = A \ r
        beta = r'*znew / (r'*z)
        p .= znew .+ beta * p;
        mul!(Ap,A,p);
        # alpha = rsold / (p' * Ap)
        alpha = r'*znew / (p'*Ap)
        x .= x .+ alpha * p;
        r .= r .- alpha * Ap;
        rsnew = r' * r
        if sqrt(rsnew) < sqrt(eps(real(eltype(b))))
              break
        end
        # p = r + (rsnew / rsold) * p;
        z .= znew
        rsold = rsnew
        @show rsold
    end
    # @show num_iter_steps
    num_iter_steps
end


function mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=4,mg_level=mg_level,use_galerkin=true)
    r = b - A * x;
    rnew = similar(r)
    # z = jacobi(A,r,maxiter=jacobi_iter)
    # mg_level = 3
    z = mg(A,r, mg_level,NUM_V_CYCLES=NUM_V_CYCLES,nu=nu,use_galerkin=use_galerkin)
    znew = similar(z)
    p = z;
    Ap = A*p;
    num_iter_steps = 0
    norms = [norm(r)]
    for step = 1:maxiter
    # for _ in 1:40
        num_iter_steps += 1
        alpha = r'*z/(p'*Ap)
        mul!(Ap,A,p);
        alpha = r'*z / (p'*Ap)
        x .= x .+ alpha * p;
        rnew .= r .- alpha * Ap;
        rsnew = rnew' * rnew
        append!(norms,sqrt(rsnew))
        if sqrt(rsnew) < abstol
              break
        end
        # p = r + (rsnew / rsold) * p;
        # znew = jacobi(A,rnew,maxiter=jacobi_iter=10);
        znew = mg(A,rnew,mg_level,NUM_V_CYCLES=NUM_V_CYCLES,nu=nu,use_galerkin=use_galerkin)
        beta = rnew'*znew/(r'*z);
        p .= znew .+ beta * p;
        z .= znew
        r .= rnew
        # @show rsnew
        # @show step, rsnew
    end
    # @show num_iter_steps
    return num_iter_steps, norms
end

function mg_preconditioned_CG_v2(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=4,mg_level=mg_level,use_galerkin=true)
    r = b - A * x;
    rnew = similar(r)
    # z = jacobi(A,r,maxiter=jacobi_iter)
    # mg_level = 3
    A_matrices = Dict()
    z = mg_v2(A,r,A_matrices,mg_level,NUM_V_CYCLES=NUM_V_CYCLES,nu=nu,use_galerkin=use_galerkin)
    znew = similar(z)
    p = z;
    Ap = A*p;
    num_iter_steps = 0
    norms = [norm(r)]
    for step = 1:maxiter
    # for _ in 1:40
        num_iter_steps += 1
        alpha = r'*z/(p'*Ap)
        mul!(Ap,A,p);
        alpha = r'*z / (p'*Ap)
        x .= x .+ alpha * p;
        rnew .= r .- alpha * Ap;
        rsnew = rnew' * rnew
        append!(norms,sqrt(rsnew))
        if sqrt(rsnew) < abstol
              break
        end
        # p = r + (rsnew / rsold) * p;
        # znew = jacobi(A,rnew,maxiter=jacobi_iter=10);
        znew = mg_v2(A,rnew,A_matrices,mg_level,NUM_V_CYCLES=NUM_V_CYCLES,nu=nu,use_galerkin=use_galerkin)
        beta = rnew'*znew/(r'*z);
        p .= znew .+ beta * p;
        z .= znew
        r .= rnew
        # @show rsnew
        # @show step, rsnew
    end
    # @show num_iter_steps
    return num_iter_steps, norms
end

function CG_CPU(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))))
    r = b - A * x;
    p = r;
    rsold = r' * r
    # Ap = spzeros(length(b))
    Ap = similar(b);

    num_iter_steps = 0
    norms = [sqrt(rsold)]
    # @show rsold
    for step = 1:maxiter
    # for _ in 1:40
        num_iter_steps += 1
        mul!(Ap,A,p);
        alpha = rsold / (p' * Ap)
        x .= x .+ alpha * p;
        r .= r .- alpha * Ap;
        rsnew = r' * r
        append!(norms,sqrt(rsnew))
        if sqrt(rsnew) < abstol
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew
        # @show rsold
        # @show step, rsold
    end
    # @show num_iter_steps
    return (num_iter_steps,norms)
end

function CG_hybrid(A,b,x;maxiter_mg=length(b),abstol=sqrt(eps(real(eltype(b)))),use_galerkin=use_galerkin,mg_cg_tol=1e-5,NUM_V_CYCLES=3,mg_level=mg_level,nu=4)
    # mg_cg_tol = 1e-4
    # mg_cg_tol = 4e-5
    iter_1, norms_mg_cg = mg_preconditioned_CG(A,b,x;maxiter=maxiter_mg,use_galerkin=use_galerkin,abstol=mg_cg_tol,NUM_V_CYCLES=NUM_V_CYCLES,mg_level=mg_level,nu=nu)
    @show iter_1
    @show norm(A*x-b)
    @show abstol
    iter_2, norms_cg = CG_CPU(A,b,x,abstol=abstol)
    @show iter_2
    # return iter_1 + iter_2
    @show norm(A*x-b)
    rsold = norm(A*x-b)^2
    # @show iter_1, iter_2
    # @show iter_1 + iter_2, rsold^2
    return iter_1, iter_2, norms_mg_cg, norms_cg
end


level=5
max_iter=(2^level+1)^2
maxiter_mg=(2^level+1)^2
reltol=sqrt(eps(real(Float64)))
repeat=1
use_galerkin=true
test_cg=false
test_jacobi=false
test_mg_cg=false
mg_cg_tol=4e-5
mg_level=3
NUM_V_CYCLES=3
nu=4

function test_preconditioned_CG(;level=5,max_iter=(2^level+1)^2,maxiter_mg=(2^level+1)^2,reltol=sqrt(eps(real(Float64))),repeat=1,use_galerkin=true,test_cg=false,test_jacobi=false,test_mg_cg=false,preassembled_A=true,mg_cg_tol=4e-5,mg_level=3,NUM_V_CYCLES=3,nu=4)
    # level = 3
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    @show size(A)
    @show size(b)
    x = zeros(Nx*Ny);
    # max_iter=length(b)
    # max_iter=100
    r_0 = norm(b-A*x)
    abstol=0
    tol = max(reltol*r_0,abstol)

    @show tol

    # repeat = 10
    @show test_cg
    @show test_jacobi
    @show test_mg_cg
    @show maxiter_mg
    @show mg_cg_tol
    @show mg_level
    @show use_galerkin
    @show preassembled_A

    if test_cg
        println()
        println("############# STARTING CG ###################")
        x = zeros(Nx*Ny);
        time_CG = @elapsed num_iter_CG = CG_CPU(A,b,x,maxiter=max_iter,abstol=tol)
        @show num_iter_CG[1], num_iter_CG[2][end]
        # time = @elapsed for _ in 1:repeat
        #     x = zeros(Nx*Ny);
        #     num_iter_CG = CG_CPU(A,b,x,maxiter=max_iter,abstol=tol)
        # end
        # time_CG = time / repeat
        println("############# END OF CG #####################")
        println()
    end
    
    if test_jacobi
        println()
        println("###### STARTING JACOBI SMOOTHED CG ##########")
        x = zeros(Nx*Ny);
        num_iter_jacobi_CG = jacobi_smoothed_CG(A,b,x,maxiter=max_iter,jacobi_iter=50,abstol=tol)
        println("######## END OF JACOBI SMOOTHED CG ##########")
        println()
        # x = zeros(Nx*Ny);
        # jacobi_preconditioned_CG(A,b,x)
    end
        
    
    if test_mg_cg
        println()
        println("###### STARTING MG PRECONDITIONED CG ##############")
        x = zeros(Nx*Ny)
        if preassembled_A
            time_mg_CG = @elapsed num_iter_mg_CG = mg_preconditioned_CG_v2(A,b,x,maxiter=max_iter,abstol=tol,nu=nu,mg_level=mg_level,use_galerkin=use_galerkin)
        else
            time_mg_CG = @elapsed num_iter_mg_CG = mg_preconditioned_CG(A,b,x,maxiter=max_iter,abstol=tol,nu=nu,mg_level=mg_level,use_galerkin=use_galerkin)
        end
        # @show num_iter_mg_CG[1], num_iter_mg_CG[2]
        @show num_iter_mg_CG[1]
        println("######## END OF MG PRECONDITIONED CG ##############")
        println()
        # time_mg_cg = @elapsed for _ in 1:repeat
        #     x = zeros(Nx*Ny)
        #     num_iter_mg_CG = mg_preconditioned_CG(A,b,x,maxiter=max_iter,abstol=tol,nu=nu,mg_level=mg_level)
        # end
        # time_mg_cg = time_mg_cg / repeat
    end
    
    println()
    println("###### STARTING HYBRID MG PRECONDITIONED CG ##############")
    x = zeros(Nx*Ny)
    time_hybrid_CG = @elapsed (num_iter_mg_CG_1, num_iter_CG_2, norms_mg_cg, norms_hybrid_cg) = CG_hybrid(A,b,x,maxiter_mg=maxiter_mg,use_galerkin=use_galerkin,abstol=tol,mg_cg_tol=mg_cg_tol,NUM_V_CYCLES=NUM_V_CYCLES,mg_level=mg_level,nu=nu)

    # time_hybrid = @elapsed for _ in 1:repeat
    #     x = zeros(Nx*Ny)
    #     (num_iter_mg_CG_1, num_iter_CG_2, norms_mg_cg, norms_cg_hybrid) = CG_hybrid(A,b,x,maxiter_mg=maxiter_mg,abstol=tol,mg_cg_tol=mg_cg_tol,NUM_V_CYCLES=NUM_V_CYCLES,mg_level=mg_level,nu=nu)
    # end
    # time_hybrid_CG = time_hybrid / repeat
    println("######## END OF HYBRID MG PRECONDITIONED CG ##############")
    println()

    @show (num_iter_mg_CG_1,num_iter_CG_2)
    # @show (norms_mg_cg, norms_hybrid_cg) 

    if test_cg
        @show num_iter_CG[1]
        time_CG_rounded = round(time_CG,digits=5)
    end

    if test_jacobi
        @show num_iter_jacobi_CG
    end

    if test_cg
        @show time_CG
    end

    if test_mg_cg
        @show time_mg_CG
        time_mg_CG_rounded = round(time_mg_CG,digits=5)
        # @show num_iter_mg_CG[1]
    end


    @show time_hybrid_CG
    # plot(log.(10,norms_hybrid_cg))
    # plot(log.(10,vcat(norms_mg_cg,norms_hybrid_cg)),label="Hybrid-CG, time=$time_hybrid_CG")
    plot()
    

    
    
    time_hybrid_CG_rounded = round(time_hybrid_CG,digits=5)

    if test_mg_cg
        plot!(log.(10,num_iter_mg_CG[2]),label="MG-CG,             time=$time_mg_CG_rounded")
    end

    if test_cg
        plot!(log.(10,num_iter_CG[2]),label="CG,                   time=$time_CG_rounded")
        # plot!(log.(10,num_iter_mg_CG[2]),label="MG-CG, time=$time_mg_CG")
    end
    plot!(log.(10,vcat(norms_mg_cg,norms_hybrid_cg)),label="Hybrid-CG,        time=$time_hybrid_CG_rounded")
end


function multigrid_iteration_matrix(level)
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    Ir = restriction_2d(Nx)
    Ip = prolongation_2d(div(Nx+1,2))
    A_2h = Ir*(A)*Ip
    part_1 = sparse(I,Nx*Ny,Nx*Ny) - Ip*(A_2h\Matrix(Ir))*A
    P = Diagonal(A)
    Q = P - A # for Jacobi
    Sm = P\Q
    part_2 = Sm
    S = part_1 * part_2
    return S
end


function multigrid_precondition_matrix(level)
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    Ir = restriction_2d(Nx)
    Ip = prolongation_2d(div(Nx+1,2))
    # A_2h = Ir*(A)*Ip
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1);
    P = Diagonal(A)
    Q = P - A # for Jacobi
    m = 4
    # m = 1
    # H = inv(Matrix(P))*Q
    H = P\Q
    dim1,dim2 = size(H)
    R = spzeros(dim1,dim2)
    for i in 0:m-1
        R += H^i * inv(P)
    end
    M_no_post = R + Ip * (A_2h \ ( Ir * (Matrix(1.0I,size(A)) - A*R)))
    M_post = H^m*R + R + H^m * Ip *( A_2h \ (Ir * (Matrix(I,size(A)) - A*R)))

    m2 = 5
    R2 = spzeros(dim1,dim2)
    for i in 0:m2-1
        R2 += H^i * inv(P)
    end
    M_post_2 = H^m2 * M_no_post + R2
    # @show eigvals(M_no_post)
    # @show eigvals(M_post)
    @show cond(Matrix(A))
    @show cond(M_no_post*Matrix(A))
    @show cond(M_post*Matrix(A))
    @show cond(M_post_2*Matrix(A))
    return M_no_post, M_post   
end




test_preconditioned_CG(level=8,test_mg_cg=true,preassembled_A=true,test_cg=true,nu=10)
test_preconditioned_CG(level=9,test_mg_cg=true,preassembled_A=true,test_cg=true,nu=10)
test_preconditioned_CG(level=10,test_mg_cg=true,preassembled_A=true,test_cg=true,nu=10)
test_preconditioned_CG(level=10,test_mg_cg=true,preassembled_A=true,test_cg=false,nu=10)
test_preconditioned_CG(level=11,test_mg_cg=true,preassembled_A=true,test_cg=false,nu=10)
# test_preconditioned_CG(level=12,test_mg_cg=true,preassembled_A=true,test_cg=false,nu=10)