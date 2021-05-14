using SparseArrays
using LinearAlgebra
using Plots
using Interpolations
using Printf
using IterativeSolvers


L = 2
k = 1
σ = 0

# Dirichlet Boundary Condition
# v0 = 1.0 vn = 1.0
# exact solution sine wave .+ 1


function exact_u(C,k,σ,x)
    return C/(π^2*k^2 + σ) * sin.(k*π*x) .+ 1
end

function f(C,k,x)
    return C*sin.(k*π*x)
end

function linear_interpolation(v)
    len_v = length(v)
    v_interpolated = zeros(2*len_v+1)
    for i in 1:2*len_v+1
        # println(i)
        if i%2 == 0
            # println("case 1")
            v_interpolated[i] = (v[div(i,2)])
        elseif i == 1 
            # println("case 2")
            # v_interpolated[i] = (v[div((i+1),2)])
            v_interpolated[i] = (v[1])/2
        elseif i == 2*len_v + 1
            v_interpolated[i] = (v[end])/2
        else
            v_interpolated[i] = (v[div(i-1,2)] + v[div(i+1,2)]) / 2
        end
    end
    return v_interpolated
end

function linear_interpolation_v2(v) # modified for v0 = 1 and vn = 1
    len_v = length(v)
    v_interpolated = zeros(2*len_v+1)
    for i in 1:2*len_v+1
        # println(i)
        if i%2 == 0
            # println("case 1")
            v_interpolated[i] = (v[div(i,2)])
        elseif i == 1 
            # println("case 2")
            # v_interpolated[i] = (v[div((i+1),2)])
            v_interpolated[i] = (v[1] + 1.0)/2
        elseif i == 2*len_v + 1
            v_interpolated[i] = (v[end] + 1.0)/2
        else
            v_interpolated[i] = (v[div(i-1,2)] + v[div(i+1,2)]) / 2
        end
    end
    return v_interpolated
end

function linear_interpolation_dirichlet(v,v0,vn) # including boundary points
    # v_interpolated = linear_interpolation(v)
    # v_0 = 1
    # v_end = 1
    # v_interpolated[1] = v_interpolated[1] + v_0/2
    # v_interpolated[end] = v_interpolated[end] + v_end/2
    # return v_interpolated
    len_v = length(v)
    v_interpolated = zeros(2*len_v - 1)
    return v_interpolated
end

function weighting(f)
    len_f = length(f)
    f_weighted = zeros(div(len_f-1,2))
    for i in 1:length(f_weighted)
        f_weighted[i] = (f[2*i-1] + 2*f[2*i] + f[2*i+1])/4
    end
    return f_weighted
end


# for one vector, linear interpolation and weighting are not necessary reverse operation
# They are associated via Galerkin property

function Jacobi_iter(ω,v,f)
    N = length(v)
    # h = v[2] - v[1]
    h = 1/(N+1)
    v_new = copy(v)
    for j = 2:N-1
        v_new[j] = (1-ω) * v[j] + ω * 1/(2 + σ*h^2) * (v[j-1] + v[j+1] + h^2*f[j])
    end
    v_new[1] = (1-ω) * v[1] + ω * 1/(2 + σ*h^2) * (v[2] + h^2*f[1]) 
    v_new[end] = (1-ω) * v[end] +  ω * 1/(2 + σ*h^2) * (v[end-1] + h^2*f[end])
    return v_new
end

function Jacobi_iter_v2(ω,v,f)
    N = length(v)
    # h = v[2] - v[1]
    h = 1/(N-1)
    v_new = copy(v)

    for j = 2:N-1
        v_new[j] = (1-ω) * v[j] + ω * 1/(2 + σ*h^2) * (v[j-1] + v[j+1] + h^2*f[j])
    end
    v_new[1] = (1-ω) * v[1] + ω * 1/(2 + σ*h^2) * (v[2] + h^2*f[1] + 1) 
    v_new[end] = (1-ω) * v[end] +  ω * 1/(2 + σ*h^2) * (v[end-1] + h^2*f[end] + 1)
    return v_new
end

function A(v)
    # h = v[2] - v[1]
    h = 1/(length(v)+1)
    v_new = similar(v)
    v_new[1] = ((2 + σ*h^2)*v[1] - v[2]) / h^2
    v_new[end] = ((2 + σ*h^2)*v[end] - v[end-1]) / h^2
    for i in 2:length(v_new)-1
        v_new[i] = (- v[i-1] + (2 + σ*h^2)*v[i] - v[i+1]) / h^2
    end
    return v_new
end




function A_matrix(N)
    A = spzeros(N,N)
    h = 1/(N+1)
    for i in 1:N
        A[i,i] = 2 + σ*h^2
    end
    for i in 1:N-1
        A[i,i+1] = -1
        A[i+1,i] = -1
    end
    return A ./ h^2
end

function A_matrix_dirichlet(N)
    A = spzeros(N,N)
    h = 1/(N-1)
    # for i in 2:N-1
    for i in 1:N
        A[i,i] = 2 + σ*h^2    
    end
    # for i in 2:N-2
    for i in 1:N-1
        A[i,i+1] = -1
        A[i+1,i] = -1
    end
    A ./= h^2
    A[1,1] = 1
    A[1,2] = 0
    A[end,end] = 1
    A[end,end-1] = 0
    return A
end

function rhs_dirichlet(N,v0,vn)
    h = 1/(N-1)
    x = 0:h:1
    # C = 1
    k = 1
    C = π^2*k^2
    rhs = C*sin.(k*π*x)
    rhs[1] = v0
    rhs[end] = vn
    return rhs
end

function V_cycle(L,iter_times,N)
    ω = 2/3
    # N = 2^7
    x = range(0,stop=1,step=1/N)
    x = x[2:end-1]
    # C = 1
    C = π^2*k^2
    # iter_times = 10
    # v = 1/2*(sin.(16*x*π) + sin.(40*x*π))
    # v = 1/2*sin.(16*x*π)
    # v = similar(x)
    v = zeros(N-1)
    # v[1] = 1
    # v[end] = 1
    # v = ones(N-1)
    # v = randn(N-1)
    rhs = C*sin.(k*π*x)
    ###
    rhs[1] += 1/(1/N)^2
    rhs[end] += 1/(1/N)^2
    ###
    v_values = Dict(1 => v)
    rhs_values = Dict(1 => rhs)
    # @show rhs_values[1]
    # @show rhs_values[1]
    for i in 1:L
        @show i
        if i != L
            for _ in 1:iter_times
                v = Jacobi_iter_v2(ω,v,rhs_values[i])
            end
            # @show v
            # v_values[i] = copy(v) # need to examine
            v_values[i] = v
            rhs = weighting(rhs_values[i] - A(v_values[i]))
            # rhs = weighting(rhs_values[i] - A_matrix(N-1)*(v_values[i]))
            rhs_values[i+1] = rhs
            N = div(N,2)
            v = zeros(N-1)
            # v[1] = 1
            # v[end] = 1
        else
            v_values[i] = A_matrix(N-1) \ rhs_values[i]
            # for _ in 1:iter_times
            #     v_values[i] = Jacobi_iter(ω,v,rhs_values[i])
            # end
        end
        @show v_values[i]
    end
    println("Pass first part")
    for i in 1:length(v_values)
        # @show length(v_values[i])
    end
    for i in 1:L-1
        j = L - i
        # @show j
        # @show v_values[j]
        # @show v_values[j+1]
        v_values[j] = v_values[j] + linear_interpolation(v_values[j+1]) # changed to Jacobi Iteration form
        v = v_values[j]
        for _ in 1:iter_times
            v = Jacobi_iter(ω,v,rhs_values[j])
        end
        v_values[j] = v
        @show j
        @show v_values[j]
    end
    return v_values[1], exact_u(C,k,σ,x)
end


# function V_cycle_kernel(vh,fh,N,L,iter_times,ω,C,x)
function V_cycle_kernel(vh,fh,L)
    N = length(vh) + 1
    x = range(0,stop=1,step=1/N)
    x = x[2:end-1]
    v_values = Dict(1=>vh)
    rhs_values = Dict(1 => fh)
    for i in 1:L
        # @show i
        if i != L
            for _ in 1:iter_times
                vh = Jacobi_iter(ω,vh,rhs_values[i])
            end
            # v_values[i] = copy(v) # need to examine
            v_values[i] = vh
            rhs = weighting(rhs_values[i] - A(v_values[i]))
            # rhs = weighting(rhs_values[i] - A_matrix(N-1)*(v_values[i]))
            rhs_values[i+1] = rhs
            N = div(N,2)
            vh = zeros(N-1)
        else
            v_values[i] = A_matrix(N-1) \ rhs_values[i]
            # for _ in 1:iter_times
            #     v_values[i] = Jacobi_iter(ω,vh,rhs_values[i])
            # end
        end
        # @show v_values[i]
    end

    # println("Pass first part")
    for i in 1:length(v_values)
        # @show length(v_values[i])
    end
    for i in 1:L-1
        j = L - i
        # @show j
        # @show v_values[j]
        # @show v_values[j+1]
        v_values[j] = v_values[j] + linear_interpolation(v_values[j+1])
        vh = v_values[j]
        for i in 1:iter_times
            vh = Jacobi_iter(ω,vh,rhs_values[j])
        end
        v_values[j] = vh
    end
    return v_values[1], exact_u(C,k,σ,x)
end

function test_V_cycle_kernel(test_times)
    global ω = 2/3
    global L = 3
    global iter_times = 1
    global C = 1
    N = 2^7
    x = range(0,stop=1,step=1/N)
    x = x[2:end-1]
 
    vh = zeros(N-1)
    rhs = C*sin.(k*π*x)
    A_matrix_form = A_matrix(N-1)

    direct_sol = A_matrix_form \ rhs

    for step in 1:test_times
        ans = V_cycle_kernel(vh,rhs,L)
        vh = ans[1]
        # err = norm(vh - exact_u(C,k,σ,x))
        err = norm(vh - direct_sol)
        # println("step: ",k," error: ",err)
        @printf "step: %6d, error: %1.15e\n" step err
    end
    # V_cycle_kernel(vh,rhs)
    direct_error = norm(direct_sol - exact_u(C,k,σ,x));
    # println("direct solve, error: ",direct_error)
    @printf "direct solve, error: %1.15e\n" direct_error
    cg_results = cg(A_matrix_form,rhs;log=true,reltol=1e-10)
    cg_error = norm(cg_results[1] - exact_u(C,k,σ,x))
    @printf "CG iterative, error: %1.15e\n " cg_error
    @printf "%s" cg_results[2]
end


global i = 1


###### Iterative Implementation of FMG, NOT SUCCESSFUL #################

# function FMG(fh)
#     global ω = 2/3
#     global iter_times = 3
#     global L = 3
#     global C = 1
#     N = length(fh) + 1
#     x = range(0,stop=1,step=1/N)
#     x = x[2:end-1]
#     vh = zeros(N-1)
#     rhs = C*sin.(k*π*x)
#     global v_values = Dict(1=>vh)
#     global rhs_values = Dict(1 => fh)
#     N_values = Dict(1=> N)
#     # global i = 1
#     global i
#     # while length(N_values[end]) > div(N,2^L) 
#     #     println(i)
#     # for i = 1:L
#     if i <= L
#         println(i)
#         if i!= L
#             rhs_values[i+1] = linear_interpolation(rhs_values[i])
#             v_values[i+1] = FMG(rhs_values[i+1])
#             N_values[i+1] = div(N_values[i],2)
#             i += 1
#         else
#             # N_values[i+1] = div(N_values[i],2)
#             vh = zeros(N_values[i]-1)
#             x = range(0,stop=1,step=1/N_values[i])
#             x = x[2:end-1]
#             v_values[i] = V_cycle_kernel(vh,rhs_values[i],L)
#         end
#     end
#     return v_values
# end


# global counter = 1


#########################################################################


"""
    FMG_test(num_v_cycles,N)
    num_v_cycles for how many v_cycles in one FMG
    N for the coarsest grid point
"""
function FMG_test(num_v_cycles,N)
    global ω = 2/3
    global iter_times = 3
    global L = 3
    global C = 1
    global counter
    ν = 3 # v cycle iter times
    # global counter = 1
    # N = length(fh) + 1
    # N = 2^3 # initial coarse grid
    N_finest = N * (2^num_v_cycles) # finest grid
    x = range(0,stop=1,step=1/N)
    x = x[2:end-1]
    x_finest = range(0,stop=1,step=1/N_finest)
    x_finest = x_finest[2:end-1]
    exact_sol_finest = exact_u(C,k,σ,x_finest)

    vh = zeros(N-1)
    rhs = C*sin.(k*π*x)
    # num_v_cycles = 3
    v_values = Dict(1=>vh)
    rhs_values = Dict(1=>rhs)
    tmp_results = Dict(1=>vh)
    for i in 1:num_v_cycles
        @show i
        for _ in 1:iter_times
            v_values[i] = Jacobi_iter(ω,v_values[i],rhs_values[i])
        end
        rhs_values[i+1] = linear_interpolation(rhs_values[i])
        tmp_results[i+1] = linear_interpolation(v_values[i])
        for _ in 1:ν
            tmp_results[i+1] = V_cycle_kernel(tmp_results[i+1],rhs_values[i+1],i+1)[1]
        end
        v_values[i+1] = tmp_results[i+1]
    end
    return v_values, exact_sol_finest
end


function plot_FMG(results)
    plot(results[2])
    plot!(results[1][length(results[1])])
end

function plot_results(results)
    plot(results[1])
    plot!(results[2])
end
