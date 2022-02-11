using LinearAlgebra
using SparseArrays
using Plots
using IterativeSolvers
include("metrics.jl")
include("transfinite_blend_metrics.jl")
include("multigrid_util.jl")

# Solving Poisson Equation with Coordinate Transformation
# Δ u(x,y) = f(x,y)
# Manufactured Solution: u(x,y) = sin(πx .+ πy) on logical grid
#                        u(x,y) = sin(πx/Lx .+ πy/Ly) on real grid

# Exact Solutions: Define vex, vex_x, vex_y, vex_xx, vex_xy, vex_yy






level = 6


function get_metrics(level;SBPp=SBPp,bc_map=bc_map,xt=xt,yt=yt)
    N = 2^level
    δNp = N + 1
    nelems = 1
    nfaces = 4
    EToN0 = zeros(Int64, 2, nelems)
    EToN0[1, :] .= N
    EToN0[2, :] .= N
    Nr = EToN0[1, :][1]
    Ns = EToN0[2, :][1]
    xt = xt
    yt = yt
    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp + Nsp
    metrics = create_metrics(SBPp,Nr,Ns,xt,yt)
    return metrics
end

function create_A_b(level)
    metrics= get_metrics(level)
    (Nr,Ns) = size(metrics.coord[1]) .- 1
    Nrp = Nr + 1
    Nsp = Ns + 1
    LFtoB = [BC_DIRICHLET,BC_DIRICHLET,BC_NEUMANN,BC_NEUMANN]
    OPTYPE = typeof(locoperator(2,8,8))

    e = 1
    lop = Dict{Int64,OPTYPE}()
    lop[e] = locoperator(SBPp,Nr,Ns,metrics,LFtoB)

    factorization = (x) -> cholesky(Symmetric(x))
    M = SBPLocalOperator1(lop,Nr,Ns,factorization)

    x_coord = metrics.coord[1]
    y_coord = metrics.coord[2]
    ge = zeros(Nrp*Nsp)
    gδe = zeros(Nrp*Nsp)
    δ = zeros(Nrp*Nsp)
    # ge = -2π^2 * u.(x_coord,y_coord')[:]


    bc_Dirichlet = (lf,x,y,e,δ) -> vex(x,y,e) # needs to be changed
    bc_Neumann   = (lf,x,y,nx,ny,e,δ) -> (nx .* vex_x(x,y,e) + ny .* vex_y(x,y,e)) # needs to be changed
    in_jump = (lf,x,y,e) -> zeros(Nrp*Nsp)
    # locbcarray_mod!(ge,lop[e],LFtoB,bc_Dirichlet,bc_Neumann,(e))

    locbcarray!(ge,gδe,lop[e],LFtoB,bc_Dirichlet,bc_Neumann,in_jump,(e, δ))

    source = (x,y,e) -> (-vex_xx(x,y,e)-vex_yy(x,y,e))
    locsourcearray!(ge,source,lop[e],e)

    return (lop[e].M̃, ge, lop[e].JH, δNp, δNp) # A, b, H_tilde
end


(A,b,H_tilde,Nx,Ny) = create_A_b(level)
(A_2h, b_2h, H_2h_tilde, Nx_2h, Ny_2h) = create_A_b(level-1)

A_matrices = Dict()

function jacobi_brittany!(x,A,b;maxiter=3, ω = 2/3)

    Pinv = Diagonal(1 ./ diag(A))
    P = Diagonal(diag(A))
    Q = A-P

    for j in 1:maxiter
        x[:] = ω * Pinv*(b .- Q*x[:]) + (1 - ω)*x[:]
    end
end

function Two_level_multigrid(A,b;nu=3,NUM_V_CYCLES=1,p=2)
    Nx = Ny = Int(sqrt(length(b)))
    level = Int(log(2,Nx-1))
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = create_A_b(level-1);

    v_values = Dict(1=>zeros(Nx*Ny))
    rhs_values = Dict(1 => b)
    N_values = Dict(1 => Nx)
    N_values[2] = div(Nx+1,2)

    x = zeros(length(b));
    v_values[1] = x
    
    for cycle_number in 1:NUM_V_CYCLES
        jacobi_brittany!(v_values[1],A,b;maxiter=nu);
        r = b - A*v_values[1];
        f = restriction_2d(Nx) * r;
        v_values[2] = A_2h \ f

        # println("Pass first part")
        e_1 = prolongation_2d(N_values[2]) * v_values[2];
        v_values[1] = v_values[1] + e_1;
        # println("After coarse grid correction, norm(A*x-b): $(norm(A*v_values[1]-b))")
        jacobi_brittany!(v_values[1],A,b;maxiter=nu);
    end
    return (v_values[1],norm(A * v_values[1] - b))
end

function precond_matrix(A, b; m=3, solver="jacobi",p=2)
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
        # wait to be implemented
    end

    for i = 1:m-1
        R += H^i * R0
    end

    # (A_2h, b_2h, x_2h, H1_2h) = get_operators(p, 2*h);
    (A_2h,b_2h,H_tilde_2h,Nx_2h,Ny_2h) = create_A_b(level-1)
    I_r = restriction_2d(Nx)
    
    I_p = prolongation_2d(Nx_2h)
    # M = H^m * (R + I_p * (A_2h\Matrix(I_r*(IN - A * R)))) + R
    M = H^m * (R - I_p * (A_2h\Matrix(I_r*(A * R - IN)))) + R
   
    return (M, R, H, I_p, A_2h, I_r, IN)
end


function mg_preconditioned_CG(A,b,x;maxiter=length(b),abstol=sqrt(eps(real(eltype(b)))),NUM_V_CYCLES=1,nu=3,use_galerkin=true,direct_sol=0,H_tilde=0)
    r = b - A * x;
    # (M, R, H, I_p, A_2h, I_r, IN) = precond_matrix(A,b;m=nu,solver="jacobi",p=p)
    # rnew = zeros(length(r))
    z = Two_level_multigrid(A,r;nu=nu,NUM_V_CYCLES=1)[1]
    # z = M*r
    p = z;
    # Ap = A*p;
    num_iter_steps = 0
    norms = [norm(r)]
    errors = []
    if direct_sol != 0 && H_tilde != 0
        append!(errors,sqrt(direct_sol' * A * direct_sol))
    end

    rzold = r'*z

    for step = 1:maxiter
    # for step = 1:100
    # for step = 1:5
        num_iter_steps += 1
        # @show norm(A*p)

        # alpha = r'*z/(p'*A*p)
        alpha = rzold / (p'*A*p)
        # @show alpha

        x .= x .+ alpha * p;

        r .= r .- alpha * A*p
        rs = r' * r
        append!(norms,sqrt(rs))
        if direct_sol != 0 && H_tilde != 0
            sol_error = sqrt((x - direct_sol)' * A * (x - direct_sol))
            # @show error
            append!(errors,sol_error)
        end
        if sqrt(rs) < abstol
            break
        end
        z = Two_level_multigrid(A,r;nu=nu,NUM_V_CYCLES=1)[1]
        # z = M*r
        rznew = r'*z
        beta = rznew/(rzold);
        p = z + beta * p;
        rzold = rznew
    end
    return num_iter_steps, norms, errors
end


function test_preconditioned_CG(;level=6)
    (A,b,H_tilde,Nx,Ny) = create_A_b(level)
    M = precond_matrix(A,b)[1]
    @show cond(Matrix(A))
    @show cond(M*A)
    x = zeros(Nx*Ny)
    mg_preconditioned_CG(A,b,x)M
end



function test_direct_solve(;level=6)
    metrics= get_metrics(level)

end


direct_sol = M.F[e] \ ge

iterative_sol = cg(lop[e].M̃,ge)

direct_sol_reshaped = reshape(direct_sol,Nrp,Nsp)

# xseries = x_coord[:,1]
# yseries = y_coord[1,:]
# plot(xseries,yseries,direct_sol_reshaped,st=:surface)

# plot(xseries,yseries,direct_sol_reshaped,st=:surface,camera=(45,45))

x_coord = metrics.coord[1]
y_coord = metrics.coord[2]
analy_sol = vex(x_coord,y_coord,e)[:]
numerical_error = sqrt((direct_sol - analy_sol)' * lop[e].JH * (direct_sol - analy_sol))
numerical_error_cg = sqrt((iterative_sol - analy_sol)' * lop[e].JH * (iterative_sol - analy_sol))
append!(errors,numerical_error)
