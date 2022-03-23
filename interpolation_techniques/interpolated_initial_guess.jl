include("Poisson_2d.jl")



(A0,b0,H_tilde_0,Nx_0,Ny_0) = Assembling_matrix(8)

sol_0 = reshape(A0\b0,Nx_0,Ny_0)


(A1,b1,H_tilde_1,Nx_1,Ny_1) = Assembling_matrix(9)
sol_1 = reshape(A1\b1,Nx_1,Ny_1)


interpolation_coeffs = zeros(Nx_1,Ny_1)

for i in 1:Nx_1
    for j in 1:Ny_1
        if i % 2 == 0 && j % 2 == 0
            interpolation_coeffs[i,j] = (sol_1[i,j] - sol_0[div(i,2)+1,div(j,2)+1]) / (-sol_0[div(i,2)+1, div(j,2)+1] + sol_0[div(i,2),div(j,2)])
        end
    end
end


function matrix_free_prolongation_2d(idata,odata)
    size_idata = size(idata)
    odata_tmp = zeros(size_idata .* 2)
    for i in 1:size_idata[1]-1
        for j in 1:size_idata[2]-1
            odata[2*i-1,2*j-1] = idata[i,j]
            odata[2*i-1,2*j] = (idata[i,j] + idata[i,j+1]) / 2
            odata[2*i,2*j-1] = (idata[i,j] + idata[i+1,j]) / 2
            odata[2*i,2*j] = (idata[i,j] + idata[i+1,j] + idata[i,j+1] + idata[i+1,j+1]) / 4
        end
    end
    for j in 1:size_idata[2]-1
        odata[end,2*j-1] = idata[end,j]
        odata[end,2*j] = (idata[end,j] + idata[end,j+1]) / 2 
    end
    for i in 1:size_idata[1]-1
        odata[2*i-1,end] = idata[i,end]
        odata[2*i,end] = (idata[i,end] + idata[i+1,end]) / 2
    end
    odata[end,end] = idata[end,end]
    return nothing
end






# Domain Size 1025 by 1025

(A2,b2,H_tilde_2,Nx_2,Ny_2) = Assembling_matrix(10)

interpolated_sol_2_initial_guess = zeros(Nx_2,Ny_2)
# obtaining initial guess via interpolation from 513 by 513
matrix_free_prolongation_2d(sol_1,interpolated_sol_2_initial_guess)

# using this interpolated initial guess 
x_with_initial_guess, history_initial_guess = cg!(interpolated_sol_2_initial_guess[:],A2,b2;abstol=norm(b2)*sqrt(eps(real(eltype(b2)))),log=true)
@show history_initial_guess.iters 
@show history_initial_guess.data[:resnorm]

# For this problem size, the number of iterations with an interpolated initial guess is 70

# Now we want to try using zero as initial guess
x_2_zero_initial_guess = zeros(Nx_2*Ny_2)
x_zero_initialization, history_zero_initialization = cg!(x_2_zero_initial_guess,A2,b2;abstol=norm(b2)*sqrt(eps(real(eltype(b2)))),log=true)

@show history_zero_initialization.iters
@show history_zero_initialization.data[:resnorm]

# For this problem size, the number of iterations with an interpolated initial guess is 2650


# Now we try two level interpolation scheme
interpolated_sol_2_two_level_interpolation = zeros(Nx_2,Ny_2)
interpolated_sol_2_one_level_interpolation = zeros(Nx_1,Ny_1)

# Using solution from 257 by 257, we interpolate our initial guess for 513 by 513
matrix_free_prolongation_2d(sol_0,interpolated_sol_2_one_level_interpolation)
# using this initial guess for the CG on 513 by 513
x_with_initial_guess_one_level, history_initial_guess_one_level = cg!(interpolated_sol_2_one_level_interpolation[:],A1,b1;abstol=norm(b1)*sqrt(eps(real(eltype(b1)))),log=true)
@show history_initial_guess_one_level.iters 
@show history_initial_guess_one_level.data[:resnorm]

# for this problem size, we need 425 iterations

# doing another interpolation using this result on 513 by 513, we get initial gess for 1025 by 1025
matrix_free_prolongation_2d(reshape(x_with_initial_guess_one_level,Nx_1,Ny_1),interpolated_sol_2_two_level_interpolation)

x_with_initial_guess_two_level, history_initial_guess_two_level = cg!(interpolated_sol_2_two_level_interpolation[:],A2,b2;abstol=norm(b2)*sqrt(eps(real(eltype(b2)))),log=true)

@show history_initial_guess_two_level.iters 
@show history_initial_guess_two_level.data[:resnorm]

# for this problem size, we need another 71 iterations, which is fairly close to interpolated it directly from an the solution on 513 by 513

# By using interpolated initial guess, we reduce the number of iterations from 2650 to 71 on a domain 1025 by 1025


# New we want to see how it works for domain 2049 by 2049

(A3,b3,H_tilde_3,Nx_3,Ny_3) = Assembling_matrix(11)
interpolated_sol_3_initial_guess = zeros(Nx_3,Ny_3)

# We interpolate from the result obtained by CG using zero initialization
matrix_free_prolongation_2d(reshape(x_zero_initialization,Nx_2,Ny_2),interpolated_sol_3_initial_guess)

x_with_initial_guess, history_initial_guess = cg!(interpolated_sol_3_initial_guess[:],A3,b3;abstol=norm(b3)*sqrt(eps(real(eltype(b3)))),log=true)

@show history_initial_guess.iters 
@show history_initial_guess.data[:resnorm]

# Using this interpolated result as the initial guess, we only need 17 iterations steps

# Now we want to see using CG with zero initialziation on a domain of 2049 by 2049
x_3_zero_initial_guess = zeros(Nx_3*Ny_3)
x_zero_initialization, history_zero_initialization = cg!(x_3_zero_initial_guess,A3,b3;abstol=norm(b3)*sqrt(eps(real(eltype(b3)))),log=true)

@show history_zero_initialization.iters
@show history_zero_initialization.data[:resnorm]

# This is extremely slow, we would need 5199 iterations
# The interpolated initial guess reduce the number of iterations from 5199 steps to just 17 steps
