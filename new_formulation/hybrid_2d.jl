using SparseArrays
using LinearAlgebra


include("diagonal_sbp.jl")



function e(i,n)
    A = Matrix{Float64}(I,n,n)
    return A[:,i]
end

function eyes(n)
    return Matrix{Float64}(I,n,n)
end

# DEFINE ANALYICAL SOLUTIONS
#

COF_x = 1;
COF_y = 1;

function analy_sol(x,y) # Defines analytical_solution
    return sin.(COF_x*π*x .+ COF_y*π*y)
end

function u_xx(x,y)
	return -COF_x^2*π^2 .*sin.(COF_x*π*x .+ COF_y*π*y)
end

function u_yy(x,y)
	return -COF_y^2*π^2 .*sin.(COF_x*π*x .+ COF_y*π*y)
end

function u_x(x,y)
	return COF_x*π .*cos.(COF_x*π.*x .+ COF_y*π*y)
end

function u_y(x,y)
	return COF_y*π .*cos.(COF_x*π .*x .+ COF_y*π*y)
end


# Util Functions


function is_symmetric(A)
    if norm(A-A')==0
        return true
    elseif norm(A-A') <= 1e-16
        println("Close enough");
        return true
    else
        return false
    end
end

function Diag(A)
    # Self defined function that is similar to Matlab Diag
    return Diagonal(A[:])
end


len_n_list = 10 # Length of n_list

n_list = Array{Int64,1}(undef,len_n_list)
for i in range(1,step=1,stop=len_n_list)
    n_list[i] = Integer(3)^(i)
end
h_list = 1 ./n_list


p = 2 # Accuracy order
i = j = 3 # Grid size level, larger value, finer grid


n_block = 3 # Number of grid along one side

n = Integer(n_list[i]) # Force n to be integer
N = n+1    # Total grid numbers, inverse of grid size plus one

function Operators_2d(i, j)
    h_list_x = h_list;
    h_list_y = h_list;

    hx = h_list_x[i];
    hy = h_list_y[j];

    n_x = Integer(n_list[i])
    n_y = Integer(n_list[j])

    # Matrix Size
    # n_x = Integer(m_list[i]);
    # n_y = Integer(n_list[j]);

    N_x = n_x + 1
    N_y = n_y + 1

    # n_x_one_third = n_x      # Integer(n_x)
    # n_y_one_third = n_y      # Integer(n_y)
    #
    # N_x_one_third = n_x_one_third + 1
    # N_y_one_third = n_x_one_third + 1

    (D1x, HIx, H1x, r1x) = diagonal_sbp_D1(p,n_x,xc=(0,1/3));
    (D2x, S0x, SNx, HI2x, H2x, r2x) = diagonal_sbp_D2(p,n_y,xc=(0,1/3));


    (D1y, HIy, H1y, r1y) = diagonal_sbp_D1(p,n_x,xc=(0,1/3));
    (D2y, S0y, SNy, HI2y, H2y, r2y) = diagonal_sbp_D2(p,n_y,xc=(0,1/3));

    BSx = sparse(SNx - S0x);
    BSy = sparse(SNy - S0y);

    Ax = BSx - H1x*D2x;
    Ay = BSy - H1y*D2y;

    A_tilde = kron(Ax,H1y) + kron(H1x,Ay);




    # Forming 2d Operators
    e_1x = sparse(e(1,N_x));
    e_Nx = sparse(e(N_x,N_y));
    e_1y = sparse(e(1,N_y));
    e_Ny = sparse(e(N_y,N_y));


    I_Nx = sparse(eyes(N_x));
    I_Ny = sparse(eyes(N_y));

    LW = sparse(kron(e_1y',I_Ny))
    LE = sparse(kron(e_Ny',I_Ny))
    LS = sparse(kron(I_Nx,e_1x'))
    LN = sparse(kron(I_Nx,e_Nx'))




    D1_x = sparse(kron(D1x,I_Ny));
    D1_y = sparse(kron(I_Nx,D1y));


    D2_x = sparse(kron(D2x,I_Ny));
    D2_y = sparse(kron(I_Nx,D2y));
    D2 = D2_x + D2_y




    HI_x = sparse(kron(HIx,I_Ny));
    HI_y = sparse(kron(I_Nx,HIy));

    H_x = sparse(kron(H1x,I_Ny));
    H_y = sparse(kron(I_Nx,H1y));

    A2_x = sparse(H_x*(kron(Ax,I_Ny)));
    A2_y = sparse(H_y*(kron(I_Nx,Ay)));

    BS_x = sparse(kron(BSx,I_Ny));
    BS_y = sparse(kron(I_Nx,BSy));


    HI_tilde = sparse(kron(HIx,HIx));
    H_tilde = sparse(kron(H1x,H1y));


    return (D1_x, D1_y, D2_x, D2_y, Ax, Ay, A_tilde, A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y , BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, LW, LE, LS, LN)
end


(D1_x, D1_y, D2_x, D2_y, Ax, Ay, A_tilde , A2_x, A2_y, D2, HI_x, HI_y, H1x, H1y, H_x, H_y, BS_x, BS_y, HI_tilde, H_tilde, I_Nx, I_Ny, LW, LE, LS, LN) = Operators_2d(i,j)


#=  Old way of creating Linear Range
# span_1_old = LinRange(0,1/3,N)
# span_2_old = LinRange(1/3,2/3,N)
# span_3_old = LinRange(2/3,1,N)
# span_old = vcat(span_1_old,span_2_old,span_3_old)
=#

normal_operators = [LW,LE,LS,LN];


n_block = 3
for span_index in 1:n_block
    @eval $ (Symbol("span_$span_index")) = $LinRange($(span_index-1)*1/n_block,$(span_index)*1/n_block,N)
end

#=
This is a demo of how to use variables created from symbolic
test_number = 1
test = @eval $ (Symbol("span_test_$test_number"))


This is equivalent to directly calling
test = span_test_1
But it gives you flexibility to call a variable inside a loop by providing the ordinal number associated to that variable


Notice: This gives you convenience at the price of performance
Since Variable creation is not expensive, it should be totally okay to do so, unless repeated variable calling might be the expensive, which still needs to be confirmed

@time @eval $ (Symbol("span_test_$i")) = $LinRange((i-1)*1/3,i*1/3,N)

@time span_test_3 = LinRange(2/3,1,N)

0.000217 seconds (83 allocations: 5.438 KiB)
0.000006 seconds (6 allocations: 224 bytes)
=#

span = Float64[]
for i in 1:n_block
    append!(span,@eval $ (Symbol("span_$i")))
end

@show span
#
# @assert span == span_old

analy_solution = analy_sol(span,span')




# SBP coefficients
γ = Dict(2=>1, 4=>0.2508560249, 6=>0.1878715026)
α = H_x[1,1]/h
σ₁ = -40
σ₂ = 1
β = 1
ϵ = 1  # Intersection
#τ = 1 # Can be set as constant
τ = 2/(h*γ[p]) + 2/(h*α)
δ_f = 0 # set to zero for there is no jump



# Generate Source Functions

for i = 1:n_block
    block_idx = i    # block index along x direction
    span_x = @eval $(Symbol("span_$i"))
    for j = 1:n_block
        block_idy = j           # block index along y direction
        span_y = @eval $(Symbol("span_$j"))
        block_id = block_idy + (block_idx - 1)*n_block   # Block index convention
        # println(span_x == span_y)    # for debugging
        @eval $ (Symbol("F_$block_id")) = $ (u_xx(span_x',span_y) .+ u_yy(span_x',span_y))
    end
end

# @eval $ (Symbol("span_$block_idx"),Symbol("span_$block_idy"))

#=   Comparison with old conventions    All confirmed

@assert F_1 == F_LB
@assert F_4 == F_MB
@assert F_5 == F_MM

@assert F_2 == F_LM
@assert F_5 == F_MM
@assert F_8 == F_RM

@assert F_3 == F_LT
@assert F_6 == F_MT
@assert F_9 == F_RT

=#


# Generate Boundary Conditions

function has_boundary(Block_idx, Block_idy,n_block)
    if (Block_idx == 1) || (Block_idx == n_block) || (Block_idy == 1) || (Block_idy == n_block)
        return true
    else
        return false
    end
end



function determine_block_type(Block_idx, Block_idy,n_block)
    if !(has_boundary(Block_idx,Block_idy,n_block))
        return 0    # Internal Block
    else
        if (Block_idx == 1 && Block_idy == 1)
            return 1 # Left Corner Block
        elseif (Block_idx == 1 && Block_idy == n_block)
            return 2
        elseif (Block_idx == n_block && Block_idy == 1)
            return 3
        elseif (Block_idx == n_block && Block_idy == n_block)
            return 4
        elseif (Block_idx == 1)
            return 5
        elseif (Block_idx == n_block)
            return 6
        elseif (Block_idy == 1)
            return 7
        elseif (Block_idy == n_block)
            return 8
        end
    end
end


# Define BC TYPES:
# 0: Dirichlet Boundary Condition
# 1: Neumann Boundary Condition
# 2: Interior Boundary Condition    # Equivalent to Dirichlet

TYPE_DIRICHLET = 0
TYPE_NEUMANN = 1

TYPE_WEST = 0
TYPE_EAST = 1
TYPE_SOUTH = 3
TYPE_NORTH = 4


#= These boundary types will be generated from a function
# TYPE_BC * 4 + TYPE_DR
# BC: Boundary condition
# DR: Direction
=#

DIRICHLET_WEST = 0
DIRICHLET_EAST = 1

#= Reserved for later use
DIRICHLET_SOUTH = 2
DIRICHLET_NORTH = 3

NEUMANN_WEST = 4
NEUMANN_EAST = 5
=#

NEUMANN_SOUTH = 6
NEUMANN_NORTH = 7


# Type of boundary
# 0: Dirichlet BC
# 1: Neumann BC
# 2: Interface


TYPE_WEST = 0
TYPE_EAST = 0
TYPE_SOUTH = 1
TYPE_NORTH = 1


function determine_boundary_type(boundary_number)
    # Need to be modified
    if (1 <= boundary_number <= n_block)
        return 0
    elseif (n_block + 1 <= boundary_number <= 2*n_block)
        return 1
    elseif (2*n_block+1 <= boundary_number <= 3*n_block)
        return 6
    elseif (3*n_block+1 <= boundary_number <= 4*n_block)
        return 7
    end
end

determine_boundary_type(1)


# function get_boundary_numbers(Block_idx,Block_idy)
#     if !(has_boundary(Block_idx,Block_idy))
#         return Int[0,0,0,0];
#     else
#         block_type = determine_block_type(Block_idx,Block_idy)
#         if (block_type == 1)
#             return [Block_idy, 2*n_block + Block_idx]
#         elseif (block_type == 2)
#             return [n_block, 3*n_block + Block_idx]
#         elseif (block_type == 3)
#             return [n_block + Block_idy, 3*n_block]
#         elseif (block_type == 4)
#             return [2*n_block, 4*n_block]
#         elseif (block_type == 5)
#             return [Block_idy]
#         elseif (block_type == 6)
#             return [n_block + Block_idy]
#         elseif (block_type == 7)
#             return [2*n_block + Block_idx]
#         elseif (block_type == 8)
#             return [3*n_block + Block_idx]
#         end
#     end
# end

function get_global_boundary(Block_idx, Block_idy, n_block)
    # We treat global interior interface the same as global boundary
    # returning global_number of [W,E,S,N]
    @assert Block_idx <= n_block
    @assert Block_idy <= n_block
    return Int[Block_idy + n_block*(Block_idx-1), Block_idy + n_block*Block_idx, n_block*(n_block+1) + Block_idx + n_block*(Block_idy-1), n_block*(n_block+1) + Block_idx + n_block*Block_idy]
end

# ########### Test global boundary ################
# n_block = 2
# for i = 1:n_block
#     for j = 1:n_block
#         @show get_global_boundary(i,j,n_block)
#     end
# end
# ################################################



# get_boundary_numbers(2,2)



function get_local_boundary(Block_idx, Block_idy,n_block)
    @assert Block_idx <= n_block
    @assert Block_idy <= n_block
    local_boundary = [2,2,2,2] # Default: All boundaries are interior boundaries
    if !(has_boundary(Block_idx,Block_idy,n_block))
        # local_boundary = [2,2,2,2]  # Order: W->E->S->N  # 2
        return local_boundary
    else
        if (Block_idx == 1)
            local_boundary[1] = TYPE_WEST;
            if (Block_idy == 1)
                local_boundary[3] = TYPE_SOUTH;
            elseif (Block_idy == n_block)
                local_boundary[4] = TYPE_NORTH;
            end
        elseif (Block_idx == n_block)
            local_boundary[2] = TYPE_EAST;
            if (Block_idy == 1)
                local_boundary[3] = TYPE_SOUTH;
            elseif (Block_idy == n_block)
                local_boundary[4] = TYPE_NORTH;
            end
        elseif (Block_idy == 1)
            local_boundary[3] = TYPE_SOUTH;
            println("yo ho")
        elseif (Block_idy == n_block)
            local_boundary[4] = TYPE_NORTH;
            println("yo ho")
        end
        return local_boundary
    end
end


get_local_boundary(3,3,3)
get_global_boundary(3,3,3)

function assemble_F(Block_idx,Block_idy,n_block)
    local_boundary = get_global_boundary(Block_idx,Block_idy,n_block)
    global_boundary = get_global_boundary(Block_idx,Block_idy,n_block)
    for i in 1:length(local_boundary)
        if (local_boundary[i] == )
end


function get_all_interfaces(n_block)
    # This function returns all interfaces for n_block by n_block block
    interfaces = Int[];
    for i = 1:n_block
        for j = 1:n_block
            local_boundary = get_local_boundary(i,j,n_block);
            global_boundary = get_global_boundary(i,j,n_block);
            for k in 1:4
                if local_boundary[k] == 2
                    if (! (global_boundary[k] ∈ interfaces))
                        append!(interfaces,global_boundary[k])
                    end
                end
            end
        end
    end
    @assert length(interfaces) == 2*n_block*(n_block-1) # Total Number of Interfaces
    return interfaces
end

# get_all_interfaces(3)
#
# get_local_boundary(2,2,2)

function get_block_LHS(local_boundary::Vector{Int64})
    # This function generate LHS for block namely M
    M_OP = - H_tilde * (D2_x + D2_y)
    # M_W = Array{Float64,2}(undef,N^2,N^2)
    # M_E = Array{Float64,2}(undef,N^2,N^2)
    # M_S = Array{Float64,2}(undef,N^2,N^2)
    # M_N = Array{Float64,2}(undef,N^2,N^2)
    M_W = spzeros(N^2,N^2)
    M_E = spzeros(N^2,N^2)
    M_S = spzeros(N^2,N^2)
    M_N = spzeros(N^2,N^2)
    if (local_boundary[1] == 0 || local_boundary[1] == 2)
        M_W .= τ*H_y*LW'*LW .- β*H_y*BS_x'*LW'*LW;
    elseif (local_boundary[1] == 2) # Interior boundary condition
        println("Neumann West Not implemented yet")
    end

    if (local_boundary[2] == 0 || local_boundary[2] == 2)
        M_E = τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE
    elseif (local_boundary[2] == 1) # Interior boundary condition
        println("Neumann West Not implemented yet")
    end

    if (local_boundary[3] == 1)
        M_S .= H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y
    elseif (local_boundary[3] == 2 || local_boundary[3] == 0)
        M_S = τ*H_x*LS'*LS - β*H_x*BS_y'*LS'*LS
    end

    if (local_boundary[4] == 1)
        M_N .= H_x*LN'*LN*BS_y - 1/τ*H_x*BS_y'*LN'*LN*BS_y
    elseif (local_boundary[4] == 2 || local_boundary[4] == 0)
        M_N .= τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN
    end
    M_OP .+= M_W .+ M_E .+ M_S .+ M_N
    # return (M_W, M_E, M_S, M_N)
    return M_OP
end



#= Generate Operators
# @assert test all confirmed

# local_operator_LB = get_local_boundary(1,1)
# M_LB_test = get_SAT_operator(local_operator_LB)
#
# local_operator_LM = get_local_boundary(1,2)
# M_LM_test = get_SAT_operator(local_operator_LM)
#
# isapprox(M_LB,M_LB_test)
# isapprox(M_LM,M_LM_test)
#
# local_operator_MM = get_local_boundary(2,2)
# M_MM_test = get_SAT_operator(local_operator_MM)
# isapprox(M_MM,M_MM_test)


# @assert isapprox(M_1,M_LB)
# @assert isapprox(M_2,M_LM)
# @assert isapprox(M_3,M_LT)
# @assert isapprox(M_4,M_MB)
#
#
# @assert isapprox(M_5,M_MM)
# @assert isapprox(M_6,M_MT)
# @assert isapprox(M_7,M_RB)
# @assert isapprox(M_8,M_RM)
# @assert isapprox(M_9,M_RT)


# M_LB_test[1] == τ*H_y*LW'*LW - β*H_y*BS_x'*LW'*LW
# M_LB_test[2] == τ*H_y*LE'*LE - β*H_y*BS_x'*LE'*LE
# M_LB_test[3] == H_x*LS'*LS*BS_y - 1/τ*H_x*BS_y'*LS'*LS*BS_y
# M_LB_test[4] == τ*H_x*LN'*LN - β*H_x*BS_y'*LN'*LN
#
# determine_block_type(1,2)
#
# n_block = 3


=#

for i in 1:n_block
    for j in 1:n_block
        local_boundary = get_local_boundary(i,j)
        block_id = j + (i-1) * n_block
         @eval $(Symbol("M_$block_id")) = $ (get_block_LHS(local_boundary))
        println(local_boundary)
    end
end

function generate_M() # This is faster
    M = M_1 # This is a nasty way to do M = blockdiag(***)
    for i in 2:n_block^2
        block_i = @eval $(Symbol("M_$i"))
        M = blockdiag(M,block_i)
        # M = copy(tmp)
    end
    return M
end
#
# function generate_M_v1()
#     M = M_1 # This is a nasty way to do M = blockdiag(***)
#     for i in 2:n_block^2
#         block_i = @eval $(Symbol("M_$i"))
#         tmp = blockdiag(M,block_i)
#         M = copy(tmp)
#     end
#     return M
# end

M = generate_M()
# M_test = generate_M()
# isapprox(M_test,M)




function get_block_RHS(local_boundary::Vector{Int64})
    b_W = spzeros(N^N,N)
    b_E = spzeros(N^N,N)
    b_S = spzeros(N^N,N)
    b_N = spzeros(N^N,N)

end


function generate_boundary_data(TYPE_WEST,TYPE_EAST,TYPE_SOUTH,TYPE_NORTH,n_block)
    # This function will generate boundary for the big block
    g_W = analy_sol(0,span)
    g_E = analy_sol(1,span)
    g_S = -u_y(span',0)
    g_N = u_y(span',0)

    return (g_W,g_E,g_S,g_N)
end

function get_local_boundary_data(Block_idx,Block_idy)
    # This function will get boundary condition for a particular block
    g_W = spzeros(N)
    g_E = spzeros(N)
    g_S = spzeros(N)
    g_N = spzeros(N)
    if !(has_boundary(Block_idx,Block_idy))
        return [g_W,g_E,g_S,g_N]
    else
        local_boundary = get_local_boundary(Block_idx,Block_idy)
        for boundary_condition = 0:1 # Find boundaries
            # TO DO FIND boundary
        end
        return [g_W,g_E,g_S,g_N]
    end
end

# To be completed
end

function get_local_F()
# To be completed
end


function get_local_lambda()
# To be completed
end


function assembly_lambda()
# To be completed
end
