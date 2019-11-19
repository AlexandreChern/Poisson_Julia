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


h_list = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]

# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
n_list = 1 ./h_list


p = 2

i = 3

h = h_list[i]

n = Integer(n_list[i])
n_half = Integer(n/2)
N = n + 1
N_half = n_half + 1

(D1, HI1, H1, r1) = diagonal_sbp_D1(p,n_half,xc=(0,0.5))
(D2, S0, SN, HI2, H2, r2) = diagonal_sbp_D2(p,n_half,xc=(0,0.5))

#span = LinRange(0,1,N)
#analy_sol = sin.(span*π)

half_span_1 = LinRange(0,0.5,N_half)
half_span_2 = LinRange(0.5,1,N_half)
span = vcat(half_span_1,half_span_2)
analy_sol = vcat(sin.(half_span_1*π),sin.(half_span_2*π))

e0 = e(1,N_half);
en = e(N_half,N_half);
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

# Still don't have a clear idea of penalty parameters

# A_u = D2 + β*HI1*BS'*e0*e0' + σ₁*HI1*e0*e0' + σ₁*HI1*en*en' + β*HI1*BS'*en*en' + ϵ*HI1*en*en'*D1
#
# A_v = D2 + σ₂*HI1*en*en'*D1 + σ₁*HI1*e0*e0' + β*HI1*BS'*e0*e0'  + ϵ*HI1*e0*e0'*D1
#
# A1_v = - σ₁*HI1*en*e0' - β*HI1*BS'*en*e0' - ϵ*HI1*en*e0'*D1  # Intersection happens to be at the maximum
#
# A2_u = - σ₁*HI1*e0*en' - β*HI1*BS'*e0*en' - ϵ*HI1*e0*en'*D1

# Without D1

A_u = D2 + β*HI1*BS'*e0*e0' + σ₁*HI1*e0*e0' + σ₁*HI1*en*en' + β*HI1*BS'*en*en' + ϵ*HI1*en*en'*BS

A_v = D2 + σ₂*HI1*en*en'*D1 + σ₁*HI1*e0*e0' + β*HI1*BS'*e0*e0'  + ϵ*HI1*e0*e0'*BS

A1_v = - σ₁*HI1*en*e0' - β*HI1*BS'*en*e0' + ϵ*HI1*en*e0'*BS  # Intersection happens to be at the maximum

A2_u = - σ₁*HI1*e0*en' - β*HI1*BS'*e0*en' + ϵ*HI1*e0*en'*BS

A = vcat(hcat(A_u,A1_v),hcat(A2_u, A_v))

# b1 = σ₁*HI1*g_L*e0 + β*HI1*BS'*g_L*e0 - 1/4*π^2*sin.(half_span_1*π/2)
# b2 = σ₂*HI1*g_R*en - 1/4*π^2*sin.(half_span_2*π/2)

b1 = σ₁*HI1*g_L*e0 + β*HI1*BS'*g_L*e0 - π^2*sin.(half_span_1*π)
b2 = σ₂*HI1*g_R*en - π^2*sin.(half_span_2*π)
b = vcat(b1,b2)

num_sol = A\b
using Plots
plot(span, num_sol)



# NOW DEFINING HYBRID METHOD parameters
F_L = -π^2*sin.(half_span_1*π)
F_R = -π^2*sin.(half_span_2*π)

L1 = e0'
L2 = en'
τ = σ₁
δ_f = 0.1

Mu =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + τ*L2'*L2 + β*L2'*L2*BS
Mv =  H1*D2 + τ*L1'*L1 + β*BS'*L1'*L1 + β*L2'*L2*BS + 1/τ*BS'*L2'*L2*BS

F = vcat(-τ*L2'-BS'*L2',-τ*L1'-BS'*L1')
F_T = hcat(-τ*L2-L2*BS, -τ*L1 - L1*BS)

D = 2*h*τ

g_bar = vcat(-τ*L1'*g_L - BS'*L1'*g_L + F_L, -L2'*g_R - 1/τ*BS'*L2'*g_R+ F_R)
g_bar_delta = 2*h*δ_f # Not Sure

Mzero = zeros(N_half,N_half)

A1 = vcat(hcat(Mu,Mzero),hcat(Mzero,Mv))

A = vcat(hcat(A1,F),hcat(F_T,D))
b = vcat(g_bar,g_bar_delta)

num_sol_2 = A\b
plot(span,num_sol_2[1:end-1])
