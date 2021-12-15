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

    (x,xr,xs) = xf(r,s)
    (y,yr,ys) = yf(r,s)

    J = xr .* ys - xs .* yr
    @assert minimum(J) > 0


    rx = ys ./ J
    sx = -yr ./ J
    ry = - xs ./ J
    sy = xr ./ J

    # variable coefficient matrix components
    crr = J .* (rx .* rx + ry .* ry)
    crs = J .* (sx .* rx + sy .* ry)
    css = J .* (sx .* sx + sy .* sy)

    #
    # Block surface matrices
    #

    (xf1, yf1) = (view(x,1,:), view(y,1,:))
    nx1 = -ys[1,:]
    ny1 = xs[1,:]
    sJ1 = hypot.(nx1,ny1)
    nx1 = nx1 ./ sJ1
    ny1 = ny1 ./ sJ1
    
    (xf2, yf2) = (view(x,Nrp,:),view(y,Nrp,:))
    nx2 = ys[end,:]
    ny2 = -xs[end,:]
    sJ2 = hypot.(nx2,ny2)
    nx2 = nx2 ./ sJ2
    ny2 = ny2 ./ sJ2

    (xf3, yf3) = (view(x,:,1),view(y,:,1))
    nx3 = yr[:,1]
    ny3 = -xr[:,1]
    sJ3 = hypot.(nx3,ny3)
    nx3 = nx3 ./ sJ3
    ny3 = ny3 ./ sJ3

    (xf4, yf4) = (view(x,:,Nsp), view(y,:,Nsp))
    nx4 = -yr[:,end]
    ny4 = xr[:,end]
    sJ4 = hypot.(nx4, ny4)
    nx4 = nx4 ./ sJ4
    ny4 = ny4 ./ sJ4

    (coord = (x,y),
        facecoord = ((xf1,xf2,xf3,xf4),(yf1,yf2,yf3,yf4)),
        crr = crr, css = css, crs = crs,
        J = J,
        sJ = (sJ1, sJ2, sJ3, sJ4),
        nx = (nx1, nx2, nx3, nx4),
        ny = (ny1, ny2, ny3, ny4),
        rx = rx, ry = ry, sx = sx, sy = sy
        )
    
end

function locoperator(p, Nr, Ns, metrics=create_metrics(p, Nr, Ns),
    LFToB=(BC_DIRICHLET, BC_DIRICHLET,
             BC_DIRICHLET, BC_DIRICHLET);
    τscale=2,
    crr=metrics.crr,
    css=metrics.css,
    crs=metrics.crs)
    csr = crs
    J = metrics.J

    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp * Nsp

    # Derivative operators for the rest of the computation
    (Dr, HrI, Hr, r) = diagonal_sbp_D1(p, Nr; xc=(-1, 1))
    Qr = Hr * Dr
    QrT = sparse(transpose(Qr))

    (Ds, HsI, Hs, s) = diagonal_sbp_D1(p, Ns; xc=(-1, 1))
    Qs = Hs * Ds
    QsT = sparse(transpose(Qs))

    # Identity matrices for the comuptation
    Ir = sparse(I, Nrp, Nrp)
    Is = sparse(I, Nsp, Nsp)

    # {{{ Set up the rr derivative matrix
    ISr0 = Array{Int64,1}(undef, 0)
    JSr0 = Array{Int64,1}(undef, 0)
    VSr0 = Array{Float64,1}(undef, 0)
    ISrN = Array{Int64,1}(undef, 0)
    JSrN = Array{Int64,1}(undef, 0)
    VSrN = Array{Float64,1}(undef, 0)

    (_, S0e, SNe, _, _, Ae, _) = variable_diagonal_sbp_D2(p, Nr, rand(Nrp))
    IArr = Array{Int64,1}(undef, Nsp * length(Ae.nzval))
    JArr = Array{Int64,1}(undef, Nsp * length(Ae.nzval))
    VArr = Array{Float64,1}(undef, Nsp * length(Ae.nzval))
    stArr = 0

    ISr0 = Array{Int64,1}(undef, Nsp * length(S0e.nzval))
    JSr0 = Array{Int64,1}(undef, Nsp * length(S0e.nzval))
    VSr0 = Array{Float64,1}(undef, Nsp * length(S0e.nzval))
    stSr0 = 0

    ISrN = Array{Int64,1}(undef, Nsp * length(SNe.nzval))
    JSrN = Array{Int64,1}(undef, Nsp * length(SNe.nzval))
    VSrN = Array{Float64,1}(undef, Nsp * length(SNe.nzval))
    stSrN = 0
    for j = 1:Nsp
    rng = (j - 1) * Nrp .+ (1:Nrp)
    (_, S0e, SNe, _, _, Ae, _) = variable_diagonal_sbp_D2(p, Nr, crr[rng])
    (Ie, Je, Ve) = findnz(Ae)
    IArr[stArr .+ (1:length(Ve))] = Ie .+ (j - 1) * Nrp
    JArr[stArr .+ (1:length(Ve))] = Je .+ (j - 1) * Nrp
    VArr[stArr .+ (1:length(Ve))] = Hs[j,j] * Ve
    stArr += length(Ve)

    (Ie, Je, Ve) = findnz(S0e)
    ISr0[stSr0 .+ (1:length(Ve))] = Ie .+ (j - 1) * Nrp
    JSr0[stSr0 .+ (1:length(Ve))] = Je .+ (j - 1) * Nrp
    VSr0[stSr0 .+ (1:length(Ve))] =  Hs[j,j] * Ve
    stSr0 += length(Ve)

    (Ie, Je, Ve) = findnz(SNe)
    ISrN[stSrN .+ (1:length(Ve))] = Ie .+ (j - 1) * Nrp
    JSrN[stSrN .+ (1:length(Ve))] = Je .+ (j - 1) * Nrp
    VSrN[stSrN .+ (1:length(Ve))] =  Hs[j,j] * Ve
    stSrN += length(Ve)
    end
    Ãrr = sparse(IArr[1:stArr], JArr[1:stArr], VArr[1:stArr], Np, Np)
    Sr0 = sparse(ISr0[1:stSr0], JSr0[1:stSr0], VSr0[1:stSr0], Np, Np)
    SrN = sparse(ISrN[1:stSrN], JSrN[1:stSrN], VSrN[1:stSrN], Np, Np)
    Sr0T = sparse(JSr0[1:stSr0], ISr0[1:stSr0], VSr0[1:stSr0], Np, Np)
    SrNT = sparse(JSrN[1:stSrN], ISrN[1:stSrN], VSrN[1:stSrN], Np, Np)
    #= affine mesh test
    # @assert Ãrr ≈ Ãrr'
    (D2, S0, SN, _, _, _) = diagonal_sbp_D2(p, Nr)
    Ar = SN - S0 - Hr * D2
    @assert Ãrr ≈ Hs ⊗ Ar =#
    # @assert Sr0 ≈ ((sparse(Diagonal(crr[1   .+ Nrp*(0:Ns)])) * Hs) ⊗ S0)
    # @assert SrN ≈ ((sparse(Diagonal(crr[Nrp .+ Nrp*(0:Ns)])) * Hs) ⊗ SN)
    # }}}

    # {{{ Set up the ss derivative matrix
    (_, S0e, SNe, _, _, Ae, _) = variable_diagonal_sbp_D2(p, Ns, rand(Nsp))
    IAss = Array{Int64,1}(undef, Nrp * length(Ae.nzval))
    JAss = Array{Int64,1}(undef, Nrp * length(Ae.nzval))
    VAss = Array{Float64,1}(undef, Nrp * length(Ae.nzval))
    stAss = 0

    ISs0 = Array{Int64,1}(undef, Nrp * length(S0e.nzval))
    JSs0 = Array{Int64,1}(undef, Nrp * length(S0e.nzval))
    VSs0 = Array{Float64,1}(undef, Nrp * length(S0e.nzval))
    stSs0 = 0

    ISsN = Array{Int64,1}(undef, Nrp * length(SNe.nzval))
    JSsN = Array{Int64,1}(undef, Nrp * length(SNe.nzval))
    VSsN = Array{Float64,1}(undef, Nrp * length(SNe.nzval))
    stSsN = 0
    for i = 1:Nrp
    rng = i .+ Nrp * (0:Ns)
    (_, S0e, SNe, _, _, Ae, _) = variable_diagonal_sbp_D2(p, Ns, css[rng])
    R = Ae - Dr' * Hr * Diagonal(css[rng]) * Dr

    (Ie, Je, Ve) = findnz(Ae)
    IAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
    JAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
    VAss[stAss .+ (1:length(Ve))] = Hr[i,i] * Ve
    stAss += length(Ve)

    (Ie, Je, Ve) = findnz(S0e)
    ISs0[stSs0 .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
    JSs0[stSs0 .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
    VSs0[stSs0 .+ (1:length(Ve))] = Hr[i,i] * Ve
    stSs0 += length(Ve)

    (Ie, Je, Ve) = findnz(SNe)
    ISsN[stSsN .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
    JSsN[stSsN .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
    VSsN[stSsN .+ (1:length(Ve))] = Hr[i,i] * Ve
    stSsN += length(Ve)
    end
    Ãss = sparse(IAss[1:stAss], JAss[1:stAss], VAss[1:stAss], Np, Np)
    Ss0 = sparse(ISs0[1:stSs0], JSs0[1:stSs0], VSs0[1:stSs0], Np, Np)
    SsN = sparse(ISsN[1:stSsN], JSsN[1:stSsN], VSsN[1:stSsN], Np, Np)
    Ss0T = sparse(JSs0[1:stSs0], ISs0[1:stSs0], VSs0[1:stSs0], Np, Np)
    SsNT = sparse(JSsN[1:stSsN], ISsN[1:stSsN], VSsN[1:stSsN], Np, Np)
    # @assert Ãss ≈ Ãss'
    #= affine mesh test
    (D2, S0, SN, _, _, _) = diagonal_sbp_D2(p, Ns)
    As = SN - S0 - Hs * D2
    @assert Ãss ≈ As ⊗ Hr =#
    # @assert Ss0 ≈ (S0 ⊗ (Hr * sparse(Diagonal(css[1:Nrp]))))
    # @assert SsN ≈ (SN ⊗ (Hr * sparse(Diagonal(css[Nrp*Ns .+ (1:Nrp)]))))
    # }}}

    # {{{ Set up the sr and rs derivative matrices
    Ãsr = (QsT ⊗ Ir) * sparse(1:length(crs), 1:length(crs), view(crs, :)) * (Is ⊗ Qr)
    Ãrs = (Is ⊗ QrT) * sparse(1:length(csr), 1:length(csr), view(csr, :)) * (Qs ⊗ Ir)
    # }}}

    Ã = Ãrr + Ãss + Ãrs + Ãsr

    #
    # Boundary point matrices
    #
    Er0 = sparse([1], [1], [1], Nrp, Nrp)
    ErN = sparse([Nrp], [Nrp], [1], Nrp, Nrp)
    Es0 = sparse([1], [1], [1], Nsp, Nsp)
    EsN = sparse([Nsp], [Nsp], [1], Nsp, Nsp)

    er0 = sparse([1  ], [1], [1], Nrp, 1)
    erN = sparse([Nrp], [1], [1], Nrp, 1)
    es0 = sparse([1  ], [1], [1], Nsp, 1)
    esN = sparse([Nsp], [1], [1], Nsp, 1)

    er0T = sparse([1], [1  ], [1], 1, Nrp)
    erNT = sparse([1], [Nrp], [1], 1, Nrp)
    es0T = sparse([1], [1  ], [1], 1, Nsp)
    esNT = sparse([1], [Nsp], [1], 1, Nsp)

    #
    # Store coefficient matrices as matrices
    #
    crs0 = sparse(Diagonal(crs[1:Nrp]))
    crsN = sparse(Diagonal(crs[Nrp * Ns .+ (1:Nrp)]))
    csr0 = sparse(Diagonal(csr[1   .+ Nrp * (0:Ns)]))
    csrN = sparse(Diagonal(csr[Nrp .+ Nrp * (0:Ns)]))

    #
    # Surface mass matrices
    #
    H1 = Hs
    H1I = HsI

    H2 = Hs
    H2I = HsI

    H3 = Hr
    H3I = HrI

    H4 = Hr
    H4I = HrI

    #
    # Penalty terms
    #
    if p == 2
    l = 2
    β = 0.363636363
    α = 1 / 2
    elseif p == 4
    l = 4
    β = 0.2505765857
    α = 17 / 48
    elseif p == 6
    l = 7
    β = 0.1878687080
    α = 13649 / 43200
    else
    error("unknown order")
    end

    ψmin = reshape((crr + css - sqrt.((crr - css).^2 + 4crs.^2)) / 2, Nrp, Nsp)
    @assert minimum(ψmin) > 0

    hr = 2 / Nr
    hs = 2 / Ns

    ψ1 = ψmin[  1, :]
    ψ2 = ψmin[Nrp, :]
    ψ3 = ψmin[:,   1]
    ψ4 = ψmin[:, Nsp]
    for k = 2:l
    ψ1 = min.(ψ1, ψmin[k, :])
    ψ2 = min.(ψ2, ψmin[Nrp + 1 - k, :])
    ψ3 = min.(ψ3, ψmin[:, k])
    ψ4 = min.(ψ4, ψmin[:, Nsp + 1 - k])
    end
    τ1 = (2τscale / hr) * (crr[  1, :].^2 / β + crs[  1, :].^2 / α) ./ ψ1
    τ2 = (2τscale / hr) * (crr[Nrp, :].^2 / β + crs[Nrp, :].^2 / α) ./ ψ2
    τ3 = (2τscale / hs) * (css[:,   1].^2 / β + crs[:,   1].^2 / α) ./ ψ3
    τ4 = (2τscale / hs) * (css[:, Nsp].^2 / β + crs[:, Nsp].^2 / α) ./ ψ4

    τ1 = sparse(1:Nsp, 1:Nsp, τ1)
    τ2 = sparse(1:Nsp, 1:Nsp, τ2)
    τ3 = sparse(1:Nrp, 1:Nrp, τ3)
    τ4 = sparse(1:Nrp, 1:Nrp, τ4)

    C̃1 =  (Sr0 + Sr0T) + ((csr0 * Qs + QsT * csr0) ⊗ Er0) + ((τ1 * H1) ⊗ Er0)
    C̃2 = -(SrN + SrNT) - ((csrN * Qs + QsT * csrN) ⊗ ErN) + ((τ2 * H2) ⊗ ErN)
    C̃3 =  (Ss0 + Ss0T) + (Es0 ⊗ (crs0 * Qr + QrT * crs0)) + (Es0 ⊗ (τ3 * H3))
    C̃4 = -(SsN + SsNT) - (EsN ⊗ (crsN * Qr + QrT * crsN)) + (EsN ⊗ (τ4 * H4))

    # TODO: Fix minus sign (reverse of the paper)
    G1 = -(Is ⊗ er0T) * Sr0 - ((csr0 * Qs) ⊗ er0T)
    G2 = + ( Is ⊗ erNT ) * SrN + ((csrN * Qs) ⊗ erNT)
    G3 = -(es0T ⊗ Ir) * Ss0 - (es0T ⊗ (crs0 * Qr))
    G4 = + ( esNT ⊗ Ir ) * SsN + (esNT ⊗ (crsN * Qr))

    F1 = G1' - ((τ1 * H1) ⊗ er0)
    F2 = G2' - ((τ2 * H2) ⊗ erN)
    F3 = G3' - (es0 ⊗ (τ3 * H3))
    F4 = G4' - (esN ⊗ (τ4 * H4))

    M̃ = Ã + C̃1 + C̃2 + C̃3 + C̃4

    # Modify the operator to handle the boundary conditions
    bctype = (BC_LOCKED_INTERFACE, BC_LOCKED_INTERFACE, BC_LOCKED_INTERFACE, BC_LOCKED_INTERFACE)
    F = (F1, F2, F3, F4)
    τ = (τ1, τ2, τ3, τ4)
    HfI = (H1I, H2I, H3I, H4I)
    # Modify operators for the BC
    for lf = 1:4
    if LFToB[lf] == BC_NEUMANN
    M̃ -= F[lf] * (Diagonal(1 ./ (diag(τ[lf]))) * HfI[lf]) * F[lf]'
    elseif !(LFToB[lf] == BC_DIRICHLET ||
    LFToB[lf] == BC_LOCKED_INTERFACE ||
    LFToB[lf] >= BC_JUMP_INTERFACE)
    error("invalid bc")
    end
    end
    bctype = (LFToB[1], LFToB[2], LFToB[3], LFToB[4])

    # (E, V) = eigen(Matrix(M̃))
    # println((minimum(E), maximum(E)))
    JH = sparse(1:Np, 1:Np, view(J, :)) * (Hs ⊗ Hr)
    (M̃ = M̃,
    F = (F1, F2, F3, F4),
    coord = metrics.coord,
    facecoord = metrics.facecoord,
    JH = JH,
    sJ = metrics.sJ,
    nx = metrics.nx,
    ny = metrics.ny,
    Hf = (H1, H2, H3, H4),
    HfI = (H1I, H2I, H3I, H4I),
    τ = (τ1, τ2, τ3, τ4),
    bctype = bctype)
end

function locbcarray!(ge, gδe, lop, LFToB, bc_Dirichlet, bc_Neumann, in_jump,
    bcargs = ())
    F = lop.F
    (xf, yf) = lop.facecoord
    Hf = lop.Hf
    sJ = lop.sJ
    nx = lop.nx
    ny = lop.ny
    τ = lop.τ
    ge[:] .= 0
    for lf = 1:4
        if LFToB[lf] == BC_DIRICHLET
            vf = bc_Dirichlet(lf, xf[lf], yf[lf], bcargs...)
        elseif LFToB[lf] == BC_NEUMANN
            gN = bc_Neumann(lf, xf[lf], yf[lf], nx[lf], ny[lf], bcargs...)
            vf = sJ[lf] .* gN ./ diag(τ[lf])
        elseif LFToB[lf] == BC_LOCKED_INTERFACE
            continue # nothing to do here
        elseif LFToB[lf] >= BC_JUMP_INTERFACE
            # In this case we need to add in half the jump
            vf = in_jump(lf, xf[lf], yf[lf], bcargs...) / 2
            gδe[lf][:] -= Hf[lf] * τ[lf] * vf
        else
            error("invalid bc")
        end
        ge[:] -= F[lf] * vf
    end
end

function locsourcearray!(ge, source, lop, volargs = ())

    (xloc, yloc) = lop.coord
    JHloc = lop.JH
    ge[:] += JHloc * source(xloc[:], yloc[:], volargs...)
  
  end

function locbcarray_mod!(ge, lop, LFToB, bc_Dirichlet, bc_Neumann,
    bcargs = ()) # probably should use this instead cuz we only have one element
    F = lop.F
    (xf, yf) = lop.facecoord
    Hf = lop.Hf
    sJ = lop.sJ
    nx = lop.nx
    ny = lop.ny
    τ = lop.τ
    ge[:] .= 0
    for lf = 1:4
        if LFToB[lf] == BC_DIRICHLET
            vf = bc_Dirichlet(lf, xf[lf], yf[lf], bcargs...)
        elseif LFToB[lf] == BC_NEUMANN
            gN = bc_Neumann(lf, xf[lf], yf[lf], nx[lf], ny[lf], bcargs...)
            vf = sJ[lf] .* gN ./ diag(τ[lf])
        elseif LFToB[lf] == BC_LOCKED_INTERFACE
            continue # nothing to do here
        else
            error("invalid bc")
        end
        ge[:] -= F[lf] * vf
    end
end

#{{{
struct SBPLocalOperator1{T<:Real, S<:Factorization}
    offset::Array{Int64,1}
    H::Array{T,1}
    X::Array{T,1}
    Y::Array{T,1}
    E::Array{Int64,1}
    F::Array{S,1}
    SBPLocalOperator1{T,S}(vstarts::Array{Int64,1}, H::Array{T,1}, X::Array{T,1},
                            Y::Array{T,1}, E::Array{Int64,1},
                            F::Array{S,1}) where {T<:Real, S<:Factorization} =
    new(vstarts, H, X, Y, E, F)
end
    
function SBPLocalOperator1(lop, Nr, Ns, factorization)
    nelems = length(lop)
    vstarts = Array{Int64, 1}(undef, nelems + 1)
    vstarts[1] = 1
    Np = Array{Int64, 1}(undef, nelems)
    VH = Array{Float64,1}(undef,0)
    X = Array{Float64,1}(undef,0)
    Y = Array{Float64,1}(undef,0)
    E = Array{Int64,1}(undef,0)
    FTYPE = typeof(factorization(sparse([1],[1],[1.0])))
    factors = Array{FTYPE, 1}(undef, nelems)
    for e = 1:nelems
        # Fill arrays to build global sparse matrix
        Np[e] = (Nr[e]+1)*(Ns[e]+1)
        vstarts[e+1] = vstarts[e] + Np[e]
    
        # Global "mass" matrix
        JH = lop[e].JH
        VH = [VH;Vector(diag(JH))]
    
        # global coordinates and element number array (needed for jump)
        (x,y) = lop[e].coord
        X = [X;x[:]]
        Y = [Y;y[:]]
        E = [E;e * ones(Int64, Np[e])]
    
        factors[e] = factorization(lop[e].M̃)
    end
    VNp = vstarts[nelems+1]-1 # total number of volume points
    
    SBPLocalOperator1{Float64, FTYPE}(vstarts, VH, X, Y, E, factors)
end
#}}}


#{{{ Transfinite Blend
function transfinite_blend(α1, α2, α3, α4, α1s, α2s, α3r, α4r, r, s)
    # +---4---+
    # |       |
    # 1       2
    # |       |
    # +---3---+
    @assert [α1(-1) α2(-1) α1( 1) α2( 1)] ≈ [α3(-1) α3( 1) α4(-1) α4( 1)]
  
  
    x = (1 .+ r) .* α2(s)/2 + (1 .- r) .* α1(s)/2 +
        (1 .+ s) .* α4(r)/2 + (1 .- s) .* α3(r)/2 -
       ((1 .+ r) .* (1 .+ s) .* α2( 1) +
        (1 .- r) .* (1 .+ s) .* α1( 1) +
        (1 .+ r) .* (1 .- s) .* α2(-1) +
        (1 .- r) .* (1 .- s) .* α1(-1)) / 4
  
    xr =  α2(s)/2 - α1(s)/2 +
          (1 .+ s) .* α4r(r)/2 + (1 .- s) .* α3r(r)/2 -
        (+(1 .+ s) .* α2( 1) +
         -(1 .+ s) .* α1( 1) +
         +(1 .- s) .* α2(-1) +
         -(1 .- s) .* α1(-1)) / 4
  
  
    xs = (1 .+ r) .* α2s(s)/2 + (1 .- r) .* α1s(s)/2 +
         α4(r)/2 - α3(r)/2 -
        (+(1 .+ r) .* α2( 1) +
         +(1 .- r) .* α1( 1) +
         -(1 .+ r) .* α2(-1) +
         -(1 .- r) .* α1(-1)) / 4
  
    return (x, xr, xs)
  end
  
  function transfinite_blend(α1, α2, α3, α4, r, s, p)
    (Nrp, Nsp) = size(r)
    (Dr, _, _, _) = diagonal_sbp_D1(p, Nrp-1; xc = (-1,1))
    (Ds, _, _, _) = diagonal_sbp_D1(p, Nsp-1; xc = (-1,1))
  
    α2s(s) = α2(s) * Ds'
    α1s(s) = α1(s) * Ds'
    α4r(s) = Dr * α4(r)
    α3r(s) = Dr * α3(r)
  
    transfinite_blend(α1, α2, α3, α4, α1s, α2s, α3r, α4r, r, s)
  end
  
  function transfinite_blend(v1::T1, v2::T2, v3::T3, v4::T4, r, s
                            ) where {T1 <: Number, T2 <: Number,
                                     T3 <: Number, T4 <: Number}
    e1(α) = v1 * (1 .- α) / 2 + v3 * (1 .+ α) / 2
    e2(α) = v2 * (1 .- α) / 2 + v4 * (1 .+ α) / 2
    e3(α) = v1 * (1 .- α) / 2 + v2 * (1 .+ α) / 2
    e4(α) = v3 * (1 .- α) / 2 + v4 * (1 .+ α) / 2
    e1α(α) = -v1 / 2 + v3 / 2
    e2α(α) = -v2 / 2 + v4 / 2
    e3α(α) = -v1 / 2 + v2 / 2
    e4α(α) = -v3 / 2 + v4 / 2
    transfinite_blend(e1, e2, e3, e4, e1α, e2α, e3α, e4α, r, s)
  end
  #}}}