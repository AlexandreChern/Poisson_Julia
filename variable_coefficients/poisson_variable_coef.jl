using LinearAlgebra
using SparseArrays
using Plots
# include("../matrix_free_new_formulation/diagonal_sbp.jl")
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

N = 2^6
δNp = N + 1
SBPp   = 2



function u(x,y)
    return sin.(π*x/Lx .+ π*y/Ly)
end


# const BC_DIRICHLET = 1
# const BC_NEUMANN = 2
# const BC_LOCKED_INTERFACE = 0
# const BC_JUMP_INTERFACE = 7

bc_map = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN,
BC_JUMP_INTERFACE]

nelems = 1
nfaces = 4

EToN0 = zeros(Int64, 2, nelems)
EToN0[1, :] .= N
EToN0[2, :] .= N



Nr = EToN0[1, :][1]
Ns = EToN0[2, :][1]

p = 2

# xf = (r,s) -> (r,ones(size(r)),zeros(size(r)))
# yf = (r,s) -> (s,zeros(size(s)),ones(size(s)))
# metrics = create_metrics(p,Nr,Ns)

el_x = 10
el_y = 10
xt = (r,s) -> (el_x .* tan.(atan((Lx )/el_x).* (0.5*r .+ 0.5))  , el_x .* sec.(atan((Lx )/el_x).* (0.5*r .+ 0.5)).^2 * atan((Lx)/el_x) * 0.5 ,zeros(size(s)))
yt = (r,s) -> (el_y .* tan.(atan((Ly )/el_y).* (0.5*s .+ 0.5))  , zeros(size(r)), el_y .* sec.(atan((Ly )/el_y).*(0.5*s .+ 0.5)) .^2 * atan((Ly )/el_y) * 0.5 )

metrics = create_metrics(p,Nr,Ns,xt,yt)

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

direct_sol = M.F[e] \ ge
direct_sol_reshaped = reshape(direct_sol,Nrp,Nsp)

xseries = x_coord[:,1]
yseries = y_coord[1,:]
plot(xseries,yseries,direct_sol_reshaped,st=:surface)

plot(xseries,yseries,direct_sol_reshaped,st=:surface,camera=(45,45))

error = sqrt((direct_sol - analy_sol)' * lop[e].JH * (direct_sol - analy_sol))