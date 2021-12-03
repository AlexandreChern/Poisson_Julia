include("../matrix_free_new_formulation/diagonal_sbp.jl")
using LinearAlgebra
using SparseArrays

const BC_DIRICHLET = 1
const BC_NEUMANN = 2
const BC_LOCKED_INTERFACE = 0
const BC_JUMP_INTERFACE = 7

⊗(A,B) = kron(A, B)

function create_metrics(pm, Nr, Ns,
                        xf = (r,s)->(r,ones(size(r)),zeros(size(r))),
                        yf = (r,s)->(s,zeros(size(s)),ones(size(s))))
    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp + Nsp

    # Derivative operators or the metric terms
    @assert pm <= 8
    pp = pm == 6 ? 8 : pm # what is this for?

    r = range(-1,stop=1,length=Nrp)
    s = range(-1,stop=1,length=Nsp)

    # Create the mesh
    r = ones(1,Nsp) ⊗ r
    s = s' ⊗ ones(Nrp)

    (x,xr,xs) = yf(r,s)
    (y,yr,ys) = yf(r,s)

    J = xr .* ys - xs .* yr
    @assert minimum(J) > 0


    rx = ys ./ J
    sx = -yr ./ J
    ry = - xs ./ J
    sy = xr ./ J

    
end