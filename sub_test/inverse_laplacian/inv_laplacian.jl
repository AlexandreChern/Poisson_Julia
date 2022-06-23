using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Plots
# pyplot()


N = 80




h = 1/(N-1)

# A = SymTridiagonal(-2*ones(N),ones(N-1))

A = spzeros(N,N)
for i in 1:N
    for j in 1:N
        if i == j
            A[i,j] = 2
        elseif abs(i-j) == 1
            A[i,j] = -1
        end
    end
end


B = zeros(size(A))
for i in 1:N
    for j in 1:N
       if i >= j
           B[i,j] = -j*(N-(i-1))/(N+1)
       else
           B[i,j] = -i*(N-(j-1))/(N+1)
       end
   end
end


B_tri = zeros(size(A))
for i in 1:N
    for j in 1:N
        if abs(i-j) <= 3
            B_tri[i,j] = B[i,j]
        end
    end
end

C= zeros(size(A))
for i in 1:N
    for j in 1:N
        if abs(i-j) == 1
            C[i,j] = 0.25
        elseif i == j
            C[i,j] = 0.75
        end
    end
end

# B_alternative = zeros(size(A))
# for i in 1:N
#     for j in 1:N
#         B_alternative[i,j] = i*(N+1-j)/(N+1)
#     end
# end

# surface(B)

identity_matrix = sparse(I,N,N)

A_x = kron(A,identity_matrix)
B_x = kron(B,identity_matrix)
C_x = kron(C,identity_matrix)

A_y = kron(identity_matrix,A)
B_y = kron(identity_matrix,B)
C_y = kron(identity_matrix,C)

A_xy = A_x + A_y
B_xy = B_x + B_y
C_xy = C_x + C_y

cond(Matrix(A_xy))
cond(Matrix(A_xy*B_xy))
cond(Matrix(A_xy*C_xy))

A_x*B_x


x = randn(N)

b = A*x

cg(A,b;log=true)

function invertible(A,b)
    x =zeros(length(b))
    for i in 1:length(x)
        sum = 0
        for j in 1:length(x)
            if i >= j
                sum += -(j)*(N+1-i)/(N+1)*b[j]
            else
                sum += -(i)*(N+1-j)/(N+1)*b[j]
            end
        end
        x[i] = sum
    end
    return x
end

invertible(A,b) - x


function analytical_solution(x,y)
    return sin.(pi.*x).*sin.(pi*y')
end


xs = Array(0:h:1)
ys = Array(0:h:1)

asol = analytical_solution(xs,ys)

function analytical_solution_1d(x)
    return sin.(pi*x)
end

function source_1d(x_loc)
    return pi^2*sin(pi*x_loc)
end

rhs = source_1d.(xs)

num_sol = A \ (rhs.*h^2)

asol_1d = analytical_solution_1d(xs)

interpolated_asol_1d = zeros(length(asol_1d)*2-1)
for i in 1:length(asol_1d)
    interpolated_asol_1d[2*i-1] = asol_1d[i]
end

# plot(interpolated_asol_1d)

xs_interpolated = Array(0:h/2:1)
h_interpolated = h/2

for i in 1:length(asol_1d)-1
    interpolated_asol_1d[2*i] = (asol_1d[i] + asol_1d[i+1]) / 2 + (source_1d(xs_interpolated[2*i])) * h_interpolated^2 / 2
end

# plot(xs_interpolated,interpolated_asol_1d)


asol_1d_interpolated = analytical_solution_1d(xs_interpolated)
error_interpolated_source = interpolated_asol_1d -  asol_1d_interpolated
norm(error_interpolated_source)


interpolated_asol_1d = zeros(length(asol_1d)*2-1)
for i in 1:length(asol_1d)
    interpolated_asol_1d[2*i-1] = asol_1d[i]
end

for i in 1:length(asol_1d)-1
    interpolated_asol_1d[2*i] = (asol_1d[i] + asol_1d[i+1]) / 2 
end
error_interpolated = interpolated_asol_1d - asol_1d_interpolated
norm(error_interpolated)