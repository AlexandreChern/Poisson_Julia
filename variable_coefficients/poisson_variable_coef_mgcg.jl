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
    # δNp = N + 1
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

function create_A_b(level;metrics=[],SBPp=SBPp)
    if length(metrics) == 0
        metrics= get_metrics(level,SBPp=SBPp)
    end
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

    return (lop[e].M̃, ge, lop[e].JH, Nrp, Nsp) # A, b, H_tilde
end


(A,b,H_tilde,Nx,Ny) = create_A_b(level)
(A_2h, b_2h, H_2h_tilde, Nx_2h, Ny_2h) = create_A_b(level-1)


function test_preconditioned_CG(;level=6,SBPp=SBPp)
    @show SBPp
    (A,b,H_tilde,Nx,Ny) = create_A_b(level,SBPp=SBPp)
    M = precond_matrix(A,b,SBPp=SBPp)[1]
    @show cond(Matrix(A))
    @show cond(M*A)
    x = zeros(Nx*Ny)
    mg_preconditioned_CG(A,b,x)
end



function test_direct_solve(;level=6,SBPp=SBPp)
    metrics= get_metrics(level,SBPp=SBPp)
    (A,b,H_tilde,Nx,Ny) = create_A_b(level,metrics=metrics,SBPp=SBPp)
    direct_sol = A\b
    x_coord = metrics.coord[1]
    y_coord = metrics.coord[2]
    e = 1
    analy_sol = vex(x_coord,y_coord,e)[:]
    numerical_error = sqrt((direct_sol - analy_sol)' * H_tilde * (direct_sol - analy_sol))
    return numerical_error
end

function test_direct_convergence(;SBPp=SBPp)
    numerical_errors = []
    for i in 4:8
        direct_error = test_direct_solve(level=i,SBPp=SBPp)
        append!(numerical_errors,direct_error)
    end
    # @show numerical_errors
    @show log.(2,numerical_errors)
end


let 
    SBPp = 2
    level = 6
    metrics= get_metrics(level,SBPp=SBPp)
    (A,b,H_tilde,Nx,Ny) = create_A_b(level,metrics=metrics,SBPp=SBPp)
    test_direct_convergence(SBPp=SBPp)
    x,history_cg = cg(A,b;log=true)
    history_cg.data[:resnorm]
    test_preconditioned_CG(level=6,SBPp=SBPp)
end
    
let 
    SBPp = 4
    level = 6
    metrics= get_metrics(level,SBPp=SBPp)
    (A,b,H_tilde,Nx,Ny) = create_A_b(level,metrics=metrics,SBPp=SBPp)
    test_direct_convergence(SBPp=SBPp)
    x,history_cg = cg(A,b,log=true)
    history_cg.data[:resnorm]
    test_preconditioned_CG(level=level,SBPp=SBPp)
end

let 
    SBPp = 6
    level = 6
    metrics= get_metrics(level,SBPp=SBPp)
    (A,b,H_tilde,Nx,Ny) = create_A_b(level,metrics=metrics)
    test_direct_convergence(SBPp=SBPp)
    x,history_cg = cg(A,b,log=true)
    history_cg.data[:resnorm]
    test_preconditioned_CG(level=level,SBPp=SBPp)
end
