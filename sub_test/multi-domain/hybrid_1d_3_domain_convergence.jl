# Solving the Poisson Equation
# We are solving the following Poisson Equation in 1D
#
# $$\left\{\begin{array}{rlrl}{u_{xx}} + \pi^2sin(\pi x) & {=0,} & {0\leq x \leq 1}  \\
# {u_{0}} & {=0,} & {x=0} \\     Dirichlet Boundary Condition
# {u}_x_{|1} & {=-\pi,} & {x=1}\end{array}\right.$$                  Neumann Boundary Condition

# Using external Julia file

include("diagonal_sbp.jl")


using LinearAlgebra
using SparseArrays
function e(i,n)
    A = Matrix{Float64}(I,n,n)
    return A[:,i]
end


# Solve for special case h = 0.05, n = 20
# Generating data

n_list = Array{Int64,1}(undef,10)
for i in range(1,step=1,stop=length(n_list))
    n_list[i] = Integer(3)^(i+1)
end

h_list = 1 ./ n_list

# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
# n_list = Int(1 ./h_list)


p = 2

i = 5




function hybrid_sbp(i)
    # NOW DEFINING HYBRID METHOD parameters
    h = h_list[i]

    n = Integer(n_list[i])
    n_one_third = Integer(n/3)
    N = n + 1
    N_one_third = n_one_third + 1

    (D1, HI1, H1, r1) = diagonal_sbp_D1(p,n_one_third,xc=(0,1/3)) # be careful about domain
    (D2, S0, SN, HI2, H2, r2) = diagonal_sbp_D2(p,n_one_third,xc=(0,1/3))

    #span = LinRange(0,1,N)
    #analy_sol = sin.(span*π)

    span_1 = LinRange(0,1/3,N_one_third)
    span_2 = LinRange(1/3,2/3,N_one_third)
    span_3 = LinRange(2/3,1,N_one_third)
    span= vcat(span_1,span_2,span_3)
    analy_sol = vcat(sin.(span_1*π),sin.(span_2*π),sin.(span_3*π))

    e0 = e(1,N_one_third);
    en = e(N_one_third,N_one_third);
    BS = SN - S0

    α = Dict(2=>1, 4=>0.2508560249, 6=>0.1878715026)
    γ = 2 # γ > 1 for stability
    #σ₁ = -γ/(α[p]*h_list[i]) # For left boundary Dirichlet condition
    #σ₂ = -γ/(α[p]*h_list[i]) # For right boundary Neumann condition
    σ₁ = -40
    σ₂ = 1
    β = 1
    ϵ = 1  # Intersection

    g_L = 0
    g_R = -π

    F_L = -π^2*sin.(span_1*π)
    F_M = -π^2*sin.(span_2*π)
    F_R = -π^2*sin.(span_3*π)

    L1 = e0'
    L2 = en'
    τ = σ₁
    δ_f = 0.1

    # G1 = L1*BS*h
    # G2 = L2*BS*h

    g_L = 0
    g_R = -π



    Mu =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + τ*L2'*L2 + β*BS'*L2'*L2
    Mv =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + τ*L2'*L2 + β*BS'*L2'*L2
    Mw =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + β*L2'*L2*BS + 1/τ*BS'*L2'*L2*BS#*BS #+G1

    # Mw = H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + β*L2'*L2*BS + 1/τ*BS'*L2'*L2*BS

    # F = vcat(τ*L2'+BS'*L2', τ*L1' + BS'*L1' + τ*L2' + BS'*L2' ,τ*L1'+ BS'*L1')
    # F_T = hcat(τ*L2+L2*BS, τ*L1 + L1*BS + τ*L2 + L2*BS  ,τ*L1 + L1*BS )
    b_zero = zeros(N_one_third)
    F = hcat(vcat(τ*L2'+BS'*L2',τ*L1'+ BS'*L1',b_zero),vcat(b_zero,τ*L2' + BS'*L2', τ*L1' + BS'*L1'))
    F_T = vcat(hcat(τ*L2+L2*BS, τ*L1 + L1*BS , b_zero'),hcat(b_zero', τ*L2 + L2*BS, τ*L1 + L1*BS))

    # D = 4*τ*h
    D = vcat(hcat(2*τ,0),hcat(0,2*τ))
    # D = vcat(hcat(τ,τ),hcat(τ,τ))

    g_bar = vcat(τ*L1'*g_L + BS'*L1'*g_L + H1*F_L, H1*F_M ,L2'*g_R + 1/τ*BS'*L2'*g_R + H1*F_R)
    # g_bar_delta = 2*h*δ_f
    g_bar_delta = vcat(2*h*δ_f,2*h*δ_f) # Not Sure

    Mzero = zeros(N_one_third,N_one_third)

    A1 = vcat(hcat(Mu,Mzero,Mzero),hcat(Mzero,Mv,Mzero),hcat(Mzero,Mzero,Mw))

    A = vcat(hcat(A1,F),hcat(F_T,D))
    b = vcat(g_bar,g_bar_delta)

    num_sol = A\b
    num_sol_tranc = num_sol[1:end-2]
    err = (num_sol_tranc - analy_sol);
    H = vcat(hcat(H1,Mzero,Mzero),hcat(Mzero,H1,Mzero),hcat(Mzero,Mzero,H1))
    normalized_err = err'*H*err
    return normalized_err
end

# test    
err = hybrid_sbp(3)


function convergence(n)
    errs = zeros(n)
    for i in range(1,stop=n)
        err = hybrid_sbp(i)
        errs[i] = err
    end
    log_errs = log.(errs)
    return log_errs
end


println(convergence(7))
