using LinearAlgebra
using SparseArrays
using Plots
using IterativeSolvers
include("metrics.jl")

# Solving Poisson Equation with Coordinate Transformation
# Δ u(x,y) = f(x,y)
# Manufactured Solution: u(x,y) = sin(πx .+ πy) on logical grid
#                        u(x,y) = sin(πx/Lx .+ πy/Ly) on real grid

# Exact Solutions: Define vex, vex_x, vex_y, vex_xx, vex_xy, vex_yy
Lx = 80
Ly = 80

(kx,ky) = (π/Lx, π/Ly)


vex(x,y,e) = begin
    if e == 1
        return sin.(kx * x .+ ky * y)
    else
        error("Not defined for multiple blocks")
    end
end

vex_x(x,y,e) = begin
    if e == 1
        return kx * cos.(kx * x .+ ky * y)
    else
        error("Not defined for multiple blocks")
    end
end

vex_y(x,y,e) = begin
    if e == 1
        return ky * cos.(kx * x .+ ky * y)
    else
        error("Not defined for multiple blocks")
    end
end

vex_xx(x, y, e) = begin
    if e == 1
        return - kx^2 * sin.(kx * x .+ ky * y)
    else
        error("invalid block")
    end
end

vex_xy(x, y, e) = begin
    if e == 1
        return - kx*ky * sin.(kx * x .+ ky * y)
    else
        error("invalid block")
    end
end

vex_yy(x, y, e) = begin
    if e == 1
        return - ky^2 * sin.(kx * x .+ ky * y)
    else
        error("invalid block")
    end
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




sim_years = 3000.

Vp = 1e-9 # plate rate
ρ = 2.670
cs = 3.464
σn = 50
RSamin = 0.01
RSamax = 0.025
RSb = 0.015
RSDc = 0.008 # change it to be 0.008
RSf0 = 0.6
RSV0 = 1e-6
RSVinit = 1e-9
RSH1 = 15
RSH2 = 18

μshear = cs^2 * ρ
η = μshear / (2 * cs)


level = 6

function create_A_b(level)
    N = 2^level

    δNp = N + 1


    SBPp = 6


    bc_map = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN,
    BC_JUMP_INTERFACE]

    nelems = 1
    nfaces = 4

    EToN0 = zeros(Int64, 2, nelems)
    EToN0[1, :] .= N
    EToN0[2, :] .= N



    Nr = EToN0[1, :][1]
    Ns = EToN0[2, :][1]



    # xf = (r,s) -> (r,ones(size(r)),zeros(size(r)))
    # yf = (r,s) -> (s,zeros(size(s)),ones(size(s)))
    # metrics = create_metrics(p,Nr,Ns)

    el_x = 10
    el_y = 10
    xt = (r,s) -> (el_x .* tan.(atan((Lx )/el_x).* (0.5*r .+ 0.5))  , el_x .* sec.(atan((Lx )/el_x).* (0.5*r .+ 0.5)).^2 * atan((Lx)/el_x) * 0.5 ,zeros(size(s)))
    yt = (r,s) -> (el_y .* tan.(atan((Ly )/el_y).* (0.5*s .+ 0.5))  , zeros(size(r)), el_y .* sec.(atan((Ly )/el_y).*(0.5*s .+ 0.5)) .^2 * atan((Ly )/el_y) * 0.5 )

    metrics = create_metrics(SBPp,Nr,Ns,xt,yt)

    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp + Nsp

    # r = range(-1,stop=1,length=Nrp)
    # s = range(-1,stop=1,length=Nsp)

    # r = ones(1,Nsp) ⊗ r
    # s = s' ⊗ ones(Nrp)

    # (x,xr,xs) = xf(r,s)
    # (y,yr,ys) = yf(r,s)

    # J = xr .* ys - xs .* yr
    # @assert minimum(J) > 0

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

    # for step = 1:maxiter
    for step = 1:100
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
    x = zeros(Nx*Ny)
    mg_preconditioned_CG(A,b,x)
end

direct_sol = M.F[e] \ ge

iterative_sol = cg(lop[e].M̃,ge)

direct_sol_reshaped = reshape(direct_sol,Nrp,Nsp)

xseries = x_coord[:,1]
yseries = y_coord[1,:]
# plot(xseries,yseries,direct_sol_reshaped,st=:surface)

# plot(xseries,yseries,direct_sol_reshaped,st=:surface,camera=(45,45))

analy_sol = vex(x_coord,y_coord,e)[:]
numerical_error = sqrt((direct_sol - analy_sol)' * lop[e].JH * (direct_sol - analy_sol))
numerical_error_cg = sqrt((iterative_sol - analy_sol)' * lop[e].JH * (iterative_sol - analy_sol))
append!(errors,numerical_error)
