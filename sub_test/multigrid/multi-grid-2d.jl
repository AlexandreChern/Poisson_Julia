using SparseArrays
using LinearAlgebra
using Plots
using Interpolations
using Printf
using IterativeSolvers


C = 1
L = 2
k = 1
l = 1
σ = 0


function exact_u(C,k,σ,x)
    return C/(π^2*k^2 + σ) * sin.(k*π*x) 
end

function exact_u_2d(C,k,l,σ,x,y)
    return  C/(π^2*k^2 + π^2*l^2 + σ) .* sin.(k*π*x) .* sin.(k*π*y)
end

function f(C,k,x)
    return C*sin.(k*π*x)
end

function f_2d(C,k,l,x,y)
    return C .* sin.(k*π*x) .* sin.(l*π*y)
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

function linear_interpolation_2d(mat)
    (dim1,dim2) = size(mat)
    mat_interpolated = zeros(2*dim1+1,2*dim2+1)
    for i in 1:2*dim1+1
        for j in 1:2*dim2 + 1
            # println(i)
            if i%2 == 0
                if j % 2 == 0
                # println("case 1")
                    mat_interpolated[i,j] = mat[div(i,2),div(j,2)]
                elseif j == 1
                    mat_interpolated[i,j] = (mat[div(i,2),1]) / 2
                elseif j == 2*dim2 + 1
                    mat_interpolated[i,j] = (mat[div(i,2),end]) / 2
                else
                    mat_interpolated[i,j] = (mat[div(i,2),div(j-1,2)] + mat[div(i,2),div(j+1,2)]) / 2
                end
            elseif i == 1 
                # println("case 2")
                # v_interpolated[i] = (v[div((i+1),2)])
                # v_interpolated[i] = (v[1])/2
                if j % 2 == 0
                    mat_interpolated[i,j] = (mat[i,div(j,2)]) / 2
                elseif j == 1
                    mat_interpolated[i,j] = (mat[i,j]) / 4
                elseif j == 2*dim2 + 1
                    mat_interpolated[i,j] = (mat[i,div(j,2)]) / 4
                else
                    mat_interpolated[i,j] = (mat[i,div(j-1,2)] + mat[i,div(j+1,2)]) / 4
                end
            elseif i == 2*dim1 + 1
                # v_interpolated[i] = (v[end])/2
                if j % 2 == 0
                    mat_interpolated[i,j] = (mat[div(i,2),div(j,2)]) / 2
                elseif j == 1
                    mat_interpolated[i,j] = (mat[div(i,2),j]) / 4
                elseif j == 2*dim2 + 1
                    mat_interpolated[i,j] = (mat[div(i,2),div(j,2)]) / 4
                else
                    mat_interpolated[i,j] = (mat[div(i,2),div(j-1,2)] + mat[div(i,2),div(j+1,2)]) / 4
                end
            else
                # v_interpolated[i] = (v[div(i-1,2)] + v[div(i+1,2)]) / 2
                if j % 2 == 0
                    mat_interpolated[i,j] = (mat[div(i-1,2),div(j,2)] + mat[div(i+1,2),div(j,2)]) / 2
                elseif j == 1
                    mat_interpolated[i,j] = (mat[div(i-1,2),j] + mat[div(i+1,2),j]) / 4
                elseif j == 2*dim2 + 1
                    mat_interpolated[i,j] = (mat[div(i-1,2),div(j,2)] + mat[div(i+1,2),div(j,2)]) / 4
                else
                    mat_interpolated[i,j] = (mat[div(i-1,2),div(j-1,2)] + mat[div(i+1,2),div(j-1,2)] + mat[div(i-1,2),div(j+1,2)] + mat[div(i+1,2),div(j+1,2)]) / 4
                end
            end
        end
    end
    return mat_interpolated
end

function weighting(f)
    len_f = length(f)
    f_weighted = zeros(div(len_f-1,2))
    for i in 1:length(f_weighted)
        f_weighted[i] = (f[2*i-1] + 2*f[2*i] + f[2*i+1])/4
    end
    return f_weighted
end


function weighting_2d(mat)
    (dim1,dim2) = size(mat)
    mat_weighted = zeros(div(dim1-1,2),div(dim2-1,2))
    for i in 1:div(dim1-1,2)
        for j in 1:div(dim2-1,2)
            mat_weighted[i,j] = (mat[2*i-1,2*j-1] + mat[2*i-1,2*i+1] + mat[2*i+1,2*j-1] + mat[2*i+1,2*j+1]
                            + 2*(mat[2*i,2*j-1] + mat[2*i,2*j+1] + mat[2*i-1,2*j] + mat[2*i+1,2*j])
                            + 4*mat[2*i,2*j]) / 16
        end
    end
    return mat_weighted
end

function Jacobi_iter(ω,v,f)
    N = length(v)
    # h = v[2] - v[1]
    h = 1/(N+1)
    v_new = copy(v)
    for j = 2:N-1
        v_new[j] = (1-ω) * v[j] + ω * 1/(2 + σ*h^2) * (v[j-1] + v[j+1] + h^2*f[j])
    end
    return v_new
end

function Jacobi_iter_2d(ω,mat,f)
    (dim1,dim2) = size(mat)
    h = 1/(dim1+1)
    mat_new = copy(mat)
    for i = 2:dim1-1
        for j = 2:dim2-1
            mat_new[i,j] = (1-ω)*mat[i,j] + ω* 1/(4 + σ*h^2) * (mat[i-1,j] + mat[i+1,j] + mat[i,j-1] + mat[i,j+1] + h^2*f[i,j])
        end
    end
    return mat_new
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

function A_2d(mat)
    (dim1,dim2) = size(mat)
    h = 1/(dim1+1)
    mat_new = copy(mat)
    mat_new[1,1] = ((4 + σ*h^2)*mat[1,1] - mat[1,2] - mat[2,1]) / h^2
    mat_new[1,dim2] = ((4 + σ*h^2)*mat[1,end] - mat[1,end-1] - mat[2,end]) / h^2
    mat_new[dim1,1] = ((4 + σ*h^2)*mat[end,1] - mat[end-1,1] - mat[end,2]) / h^2
    mat_new[dim1,dim2] = ((4 + σ*h^2)*mat[dim1,dim2] - mat[dim1-1,dim2] - mat[dim1,dim2-1]) / h^2
    for i in 2:dim1-1
        mat_new[i,1] = ((4 + σ*h^2)*mat[i,1] - mat[i-1,1] - mat[i+1,1] - mat[i,2]) / h^2
        mat_new[i,end] = ((4 + σ*h^2)*mat[i,end] - mat[i-1,end] - mat[i+1,end] - mat[i,end-1]) / h^2
    end
    # for i in 2:dim1-1
    #     mat_new[i,end] = ((4 + σ*h^2)*mat[i,end] - mat[i-1,end] - mat[i+1,end] - mat[i,end-1]) / h^2
    # end
    for j in 2:dim1-1
        mat_new[1,j] = ((4 + σ*h^2)*mat[1,j] - mat[1,j-1] - mat[1,j+1] - mat[2,j]) / h^2
        mat_new[end,j] = ((4 + σ*h^2)*mat[end,j] - mat[end,j-1] - mat[end,j+1] - mat[end-1,j]) / h^2
    end

    for i in 2:dim1-1
        for j in 2:dim2-1
            mat_new[i,j] = ((4 + σ*h^2)*mat[i,j] - mat[i-1,j] - mat[i+1,j] - mat[i,j-1] - mat[i,j+1]) / h^2
        end
    end
    return mat_new
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

function A_matrix_2d(N)
    sparse_I = sparse(I,N,N)
    A_matrix_1d = A_matrix(N)
    return kron(A_matrix_1d,sparse_I) + kron(sparse_I,A_matrix_1d)
end

function V_cycle_kernel(vh,fh)
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


function V_cycle_kernel_2d(vh,fh)
    # N = length(vh) + 1
    (dim1,dim2) = size(vh)
    x = range(0,stop=1,step=1/(dim1+1))
    x = x[2:end-1]
    y = range(0,stop=1,step=1/(dim2+1))
    y = y[2:end-1]
    v_values = Dict(1=>vh)
    rhs_values = Dict(1 => fh)
    for i in 1:L
        # @show i
        if i != L
            for _ in 1:iter_times
                vh = Jacobi_iter_2d(ω,vh,rhs_values[i])
            end
            # v_values[i] = copy(v) # need to examine
            v_values[i] = vh
            rhs = weighting_2d(rhs_values[i] - A(v_values[i]))
            # rhs = weighting(rhs_values[i] - A_matrix(N-1)*(v_values[i]))
            rhs_values[i+1] = rhs
            # N = div(N,2)
            # vh = zeros(N-1)
            (dim1,dim2) = div.((dim1-1,dim2-1),2)
            vh = zeros(dim1,dim2)
            # @show rhs_values[i+1]
        else
            # @show A_matrix_2d(dim1)
            # @show rhs_values[i][:]
            # rhs_coarse = rhs_values[i]
            # v_values[i] = A_matrix_2d(dim1) \ (rhs_coarse[:])
            # @show A_matrix_2d(dim1) \ rhs_values[i][:]
            # for _ in 1:iter_times
            #     v_values[i] = Jacobi_iter(ω,vh,rhs_values[i])
            # end
            v_values[i] = reshape(A_matrix_2d(dim1) \ rhs_values[i][:],dim1,dim2)
        end
        # @show v_values[i]
    end

    v_values_backup = v_values

    # println("Pass first part")
    for i in 1:length(v_values)
        # @show length(v_values[i])
        # @show i, length(v_values[i])
    end
    for i in 1:L-1
        j = L - i
        # @show j
        # @show v_values[j]
        # @show v_values[j+1]
        v_values[j] = v_values[j] .+ linear_interpolation_2d(v_values[j+1])
        vh = v_values[j]
        for i in 1:iter_times
            vh = Jacobi_iter_2d(ω,vh,rhs_values[j])
        end
        v_values[j] = vh
    end
    return v_values[1], exact_u_2d(C,k,l,σ,x,y'), v_values_backup
end

function test_V_cycle_kernel(test_times)
    global ω = 2/3
    global L = 3
    global iter_times = 3
    global C = 1
    N = 2^7
    x = range(0,stop=1,step=1/N)
    x = x[2:end-1]
 
    vh = zeros(N-1)
    rhs = C*sin.(k*π*x)
    A_matrix_form = A_matrix(N-1)

    direct_sol = A_matrix_form \ rhs

    for step in 1:test_times
        ans = V_cycle_kernel(vh,rhs)
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


function test_V_cycle_kernel_2d()
    global ω = 4/5
    global L = 2
    global iter_times = 100
    global C = 1
    N = 2^4
    x = range(0,stop=1,step=1/N)
    x = x[2:end-1]
    y = range(0,stop=1,step=1/N)
    y = y[2:end-1]

    vh = zeros(N-1,N-1)
    rhs = f_2d(C,k,l,x,y')
    A_matrix_form_2d = A_matrix_2d(N-1)
    direct_sol = A_matrix_form_2d \ rhs[:]
    direct_sol = reshape(direct_sol,N-1,N-1)
    plot(x,y,direct_sol,st=:surface)
    (v_cycle_2d, exact_2d, v_values_backup) = V_cycle_kernel_2d(vh,rhs)
    plot(x,y,exact_2d,st=:surface)
    plot(x,y,v_cycle_2d,st=:surface)
end