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
    # restriction_1d = restriction_matrix_normal(N)
    restriction_1d = restriction_matrix(N)
    restriction_2d = kron(restriction_1d,restriction_1d)
    return restriction_2d
end


function CG_CPU(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),direct_sol=0,H_tilde=0)
    r = b - A * x;
    p = r;
    rsold = r' * r
    # Ap = spzeros(length(b))
    Ap = similar(b);

    num_iter_steps = 0
    norms = [sqrt(rsold)]
    errors = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol))
    end
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
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol))
            append!(errors,error)
        end
        if sqrt(rsnew) < abstol
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew
        # @show rsold
        # @show step, rsold
    end
    # @show num_iter_steps
    return (num_iter_steps,norms,errors)
end

function jacobi_iter!(x,A,b;maxiter=4,ω=2/3)
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
            x[i] = (ω) * (b[i] - σ)/A[i,i] + (1-ω)*x[i]
        end
    end
    # (x,iter_times)
end

function jacobi_brittany!(x,A,b;maxiter=3, ω = 2/3)

    Pinv = Diagonal(1 ./ diag(A))
    P = Diagonal(diag(A))
    Q = A-P

    for j in 1:maxiter
        x[:] = ω * Pinv*(b .- Q*x[:]) + (1 - ω)*x[:]
    end
    
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

function Two_level_multigrid(A,b,A_matrices;nu=3,NUM_V_CYCLES=1,use_galerkin=true)
    # level = 8
    # (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    
    

    # L = 3 # multigrid level
    Nx = Ny = Int(sqrt(length(b)))
    level = Int(log(2,Nx-1))
    (A_p,b_p,H_tilde_p,Nx_p,Ny_p) = Assembling_matrix(level-1);

    v_values = Dict(1=>zeros(Nx*Ny))
    rhs_values = Dict(1 => b)
    N_values = Dict(1 => Nx)
    N_values[2] = div(Nx+1,2)

    x = zeros(length(b));
    v_values[1] = x

    # @show A_matrices
    # @show isempty(A_matrices)
    if isempty(A_matrices)
        A_matrices[1] = A
        # N_values = Dict(1=>Nx)
        # NUM_V_CYCLES = 2
        for cycle_number in 1:NUM_V_CYCLES
            # @show cycle_number
            # max_iter = 10
            # jacobi!(v_values[1],A,b;maxiter=nu);
            jacobi_brittany!(v_values[1],A,b;maxiter=nu);
            r = b - A_matrices[1]*v_values[1];
            f = restriction_2d(Nx) * r;
            A_coarse = restriction_2d(Nx) * A_matrices[1] * prolongation_2d(N_values[2]);
            A_coarse = A_p

            if use_galerkin
                # x_p = A_coarse \ f
                A_matrices[2] = lu(A_coarse)
                v_values[2] = A_matrices[2] \ f
            else
                # x_p = A_p \ f
                A_matrices[2] = lu(Assembling_matrix(level-1)[1]);
                v_values[2] = x_p
            end 
            # println("Pass first part")
            e_1 = prolongation_2d(N_values[2]) * v_values[2];
            v_values[1] = v_values[1] + e_1;
            # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
            # jacobi!(v_values[1],A_matrices[1],b;maxiter=nu);
            jacobi_brittany!(v_values[1],A_matrices[1],b;maxiter=nu);
            # @show norm(A_matrices[1] * v_values[1] - b)
        end
        return (v_values[1],norm(A_matrices[1] * v_values[1] - b))
    else
        for cycle_number in 1:NUM_V_CYCLES
            # @show cycle_number
            # max_iter = 10
            jacobi_brittany!(v_values[1],A,b;maxiter=nu);
            # jacobi!(v_values[1],A,b;maxiter=nu);
            r = b - A*v_values[1];
            f = restriction_2d(Nx) * r;
            v_values[2] = A_matrices[2] \ f

            # println("Pass first part")
            e_1 = prolongation_2d(N_values[2]) * v_values[2];
            v_values[1] = v_values[1] + e_1;
            # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
            jacobi_brittany!(v_values[1],A_matrices[1],b;maxiter=nu);
            # jacobi!(v_values[1],A_matrices[1],b;maxiter=0);
            # @show norm(A_matrices[1] * v_values[1] - b)
        end
        return (v_values[1],norm(A_matrices[1] * v_values[1] - b))
        # return (v_values[1],norm(A_matrices[1] * v_values[1] - b) / (N_values[1] - 1))
    end
end


function mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=4,use_galerkin=true,direct_sol=0,H_tilde=0)
    r = b - A * x;
    rnew = similar(r)
    A_matrices = Dict()
    z = Two_level_multigrid(A,r,A_matrices;nu=nu,NUM_V_CYCLES=1,use_galerkin=true)[1]
    znew = similar(z)
    p = z;
    Ap = A*p;
    num_iter_steps = 0
    norms = [norm(r)]
    errors = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol))
    end

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
        if direct_sol != 0 && H_tilde != 0
            error = sqrt((x - direct_sol)' * A * (x - direct_sol))
            append!(errors,error)
        end
        if sqrt(rsnew) < abstol
            break
        end
        # p = r + (rsnew / rsold) * p;
        # znew = jacobi(A,rnew,maxiter=jacobi_iter=10);
        znew = Two_level_multigrid(A,rnew,A_matrices;nu=nu,NUM_V_CYCLES=1,use_galerkin=true)[1]
        beta = rnew'*znew/(r'*z);
        p .= znew .+ beta * p;
        z .= znew
        r .= rnew
        # @show rsnew
        # @show step, rsnew
    end
    # @show num_iter_steps
    return num_iter_steps, norms, errors
end

# function multigrid_precondition_matrix(level;m=4)
#     (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
#     Ir = restriction_2d(Nx)
#     Ip = prolongation_2d(div(Nx+1,2))
#     # A_2h = Ir*(A)*Ip
#     (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1);
#     P = Diagonal(A)
#     Q = P - A # for Jacobi
#     # m = 4
#     # m = 1
#     # H = inv(Matrix(P))*Q
#     H = P\Q
#     dim1,dim2 = size(H)
#     R = spzeros(dim1,dim2)
#     for i in 0:m-1
#         R += H^i * inv(P)
#     end
#     M_no_post = R + Ip * (A_2h \ ( Ir * (Matrix(1.0I,size(A)) - A*R)))
#     M_post = H^m*R + R + H^m * Ip *( A_2h \ (Ir * (Matrix(I,size(A)) - A*R)))

#     m2 = m-1
#     R2 = spzeros(dim1,dim2)
#     for i in 0:m2-1
#         R2 += H^i * inv(P)
#     end
#     M_post_2 = H^m2 * M_no_post + R2
#     # @show eigvals(M_no_post)
#     # @show eigvals(M_post)
#     @show cond(Matrix(A))
#     @show cond(M_no_post*Matrix(A))
#     @show cond(M_post*Matrix(A))
#     @show cond(M_post_2*Matrix(A))
#     return M_no_post, M_post, M_post_2   
# end

function precond_matrix(A, b; m=3, solver="jacobi")
    #pre and post smoothing 
    N = length(b)
    Nx = Ny = Integer((sqrt(N)))
    level = Integer(log(2,Nx-1))
    IN = sparse(Matrix(I, N, N))
    P = Diagonal(diag(A))
    Pinv = Diagonal(1 ./ diag(A))
    Q = P-A
    L = A - triu(A)
    U = A - tril(A)

    if solver == "jacobi"
       ω = 2/3
        H = ω*Pinv*Q + (1-ω)*IN 
        R = ω*Pinv 
        R0 = ω*Pinv 
    elseif solver == "ssor"
        ω = 1.4  #this is just a guess. Need to compute ω_optimal (from jacobi method)
        B1 = (P + ω*U)\Matrix(-ω*L + (1-ω)*P)
        B2 = (P + ω*L)\Matrix(-ω*U + (1-ω)*P) 
        H = B1*B2
        X = (P+ω*L)\Matrix(IN)
   
        R = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
        R0 = ω*(2-ω)*(P+ω*U)\Matrix(P*X)
    else   
    end

    for i = 1:m-1
        R += H^i * R0
    end

    # (A_2h, b_2h, x_2h, H1_2h) = get_operators(p, 2*h);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = Assembling_matrix(level-1)
    I_r = restriction_2d(Nx)
    
    I_p = prolongation_2d(Nx_2h)
    # M = H^m * (R + I_p * (A_2h\Matrix(I_r*(IN - A * R)))) + R
    M = H^m * (R - I_p * (A_2h\Matrix(I_r*(A * R - IN)))) + R
   
    return (M, R, H, I_p, A_2h, I_r, IN)
end

function test_preconditioned_CG(;level=6,nu=3,ω=2/3)
    (A,b,H_tilde,Nx,Ny) = Assembling_matrix(level);
    (M, R, H, I_p, A_2h, I_r, IN) = precond_matrix(A,b;m=nu,solver="jacobi")
    cond_A = cond(Array(A))

    # M_no_post, M_post, M_post_2 = multigrid_precondition_matrix(level;m=nu)
    # plot(eigvals(Matrix(M_no_post*A)))
    # savefig("eigvals_M_no_post_A.png")

    # plot(eigvals(Matrix(M*A)))
    # savefig("eigvals_M_post_A.png")

    # plot(eigvals(Matrix(M_post_2*A)))
    # savefig("eigvals_M_post_2_A.png")
    cond_A_M = cond(M * A)

    direct_sol = A\b
    reltol = sqrt(eps(real(eltype(b))))
    x = zeros(Nx*Ny);
    abstol = norm(A*x-b) * reltol


    x = zeros(Nx*Ny);
    iter_cg, norm_cg, errors = CG_CPU(A,b,x;maxiter=length(b),abstol=abstol)

    x = zeros(Nx*Ny)
    iter_cg, norm_cg, error_cg = CG_CPU(A,b,x;maxiter=length(b),abstol=abstol,direct_sol=direct_sol,H_tilde=H_tilde)
    
    error_cg_bound_coef = (sqrt(cond_A) - 1) / (sqrt(cond_A) + 1)
    error_cg_bound = error_cg[1] .* 2 .* error_cg_bound_coef .^ (0:1:length(error_cg)-1)


    x = zeros(Nx*Ny);
    iter_mg_cg, norm_mg_cg, error_mg_cg = mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true,direct_sol=direct_sol,H_tilde=H_tilde)
    error_mg_cg_bound_coef = (sqrt(cond_A_M) - 1) / (sqrt(cond_A_M) + 1)
    error_mg_cg_bound = error_cg[1] .* 2 .* error_mg_cg_bound_coef .^ (0:1:length(error_mg_cg)-1)

    plot(error_cg,label="error_cg")
    plot!(error_cg_bound, label="error_cg_bound")
    plot!(error_mg_cg,label="error_mg_cg")
    plot!(error_mg_cg_bound,label="error_mg_cg_bound")
    savefig("plot1.png")

    plot(log.(10,error_cg),label="error_cg")
    plot!(log.(10,error_cg_bound), label="error_cg_bound")
    plot!(log.(10,error_mg_cg),label="error_mg_cg")
    plot!(log.(10,error_mg_cg_bound),label="error_mg_cg_bound")
    savefig("plot2.png")

    time_mg_cg = @elapsed for _ in 1:2
        x = zeros(Nx*Ny)
        mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=abstol,NUM_V_CYCLES=1,nu=nu,use_galerkin=true)
    end

    time_cg = @elapsed for _ in 1:2
        x = zeros(Nx*Ny)
        CG_CPU(A,b,x;maxiter=length(b),abstol=abstol)
    end
    @show iter_cg, norm_cg[end], time_cg
    @show iter_mg_cg, norm_mg_cg[end], time_mg_cg
    # @show time_cg
    # @show time_mg_cg
    nothing
end