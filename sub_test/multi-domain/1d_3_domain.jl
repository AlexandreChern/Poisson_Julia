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

n_list = Array{Int64,1}(undef,6)
for i in range(1,step=1,stop=6)
    n_list[i] = Integer(3)^(i+1)
end

h_list = 1 ./ n_list

# h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125] # uncomment to use for p = 4, 6, 8
# n_list = Int(1 ./h_list)


p = 2

i = 5

h = h_list[i]

n = Integer(n_list[i])
n_one_third = Integer(n/3)
N = n + 1
N_one_third = n_one_third + 1

(D1, HI1, H1, r1) = diagonal_sbp_D1(p,n_one_third,xc=(0,1/3))
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


# u for left one third
A_u = D2 + β*HI1*BS'*e0*e0' + σ₁*HI1*e0*e0' + σ₁*HI1*en*en' + β*HI1*BS'*en*en' + ϵ*HI1*en*en'*BS
# v for middle one third
A_v = D2 + β*HI1*BS'*e0*e0' + σ₁*HI1*e0*e0' + ϵ*HI1*e0*e0'*BS + β*HI1*BS'*en*en' + σ₁*HI1*en*en' + ϵ*HI1*en*en'*BS
# w for right one third
A_w = D2 + σ₂*HI1*en*en'*D1 + σ₁*HI1*e0*e0' + β*HI1*BS'*e0*e0'  + ϵ*HI1*e0*e0'*BS

# Off diagonal terms

A1_v = - σ₁*HI1*en*e0' - β*HI1*BS'*en*e0' + ϵ*HI1*en*e0'*BS  # Intersection happens to be at the maximum

A2_u = - σ₁*HI1*e0*en' - β*HI1*BS'*e0*en' + ϵ*HI1*e0*en'*BS

A2_w = - σ₁*HI1*en*e0' - β*HI1*BS'*en*e0' + ϵ*HI1*en*e0'*BS

A3_v = - σ₁*HI1*e0*en' - β*HI1*BS'*e0*en' + ϵ*HI1*e0*en'*BS

A_zero = zeros(N_one_third,N_one_third);

A = vcat(hcat(A_u,A1_v,A_zero),hcat(A2_u, A_v, A2_w),hcat(A_zero, A3_v, A_w))

# b1 = σ₁*HI1*g_L*e0 + β*HI1*BS'*g_L*e0 - 1/4*π^2*sin.(half_span_1*π/2)
# b2 = σ₂*HI1*g_R*en - 1/4*π^2*sin.(half_span_2*π/2)

b1 = σ₁*HI1*g_L*e0 + β*HI1*BS'*g_L*e0 - π^2*sin.(span_1*π)
b2 = zeros(N_one_third) - π^2*sin.(span_2*π)
b3 = σ₂*HI1*g_R*en - π^2*sin.(span_3*π)
b = vcat(b1,b2,b3)
b_zero = zeros(N_one_third)

num_sol = A\b
using Plots
plot(span, num_sol)
