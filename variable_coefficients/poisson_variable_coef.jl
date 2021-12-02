using LinearAlgebra
using SparseArrays
using SparseArrays
using Plots
include("../matrix_free_new_formulation/diagonal_sbp.jl")

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

N = 400
δNp = N + 1
SBPp   = 2


const BC_DIRICHLET = 1
const BC_NEUMANN = 2
const BC_LOCKED_INTERFACE = 0
const BC_JUMP_INTERFACE = 7

bc_map = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN,
BC_JUMP_INTERFACE]

nelems = 1
nfaces = 4

EToN0 = zeros(Int64, 2, nelems)
EToN0[1, :] .= N
EToN0[2, :] .= N

Lx = 80
Ly = 80

Nr = EToN0[1, :]
Ns = EToN0[2, :]

