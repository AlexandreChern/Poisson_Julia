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



# using this interpolated initial guess 


(A2,b2,H_tilde_2,Nx_2,Ny_2) = Assembling_matrix(10)

interpolated_sol_2_initial_guess = zeros(Nx_2,Ny_2)
matrix_free_prolongation_2d(sol_1,interpolated_sol_2_initial_guess)


x_with_initial_guess, history_initial_guess = cg!(interpolated_sol_2_initial_guess[:],A2,b2;abstol=norm(b2)*sqrt(eps(real(eltype(b2)))),log=true)

@show history_initial_guess.iters 
@show history_initial_guess.data[:resnorm]


x_2_zero_initial_guess = zeros(Nx_2*Ny_2)
x_zero_initialization, history_zero_initialization = cg!(x_2_zero_initial_guess,A2,b2;abstol=norm(b2)*sqrt(eps(real(eltype(b2)))),log=true)

@show history_zero_initialization.iters
@show history_zero_initialization.data[:resnorm]


interpolated_sol_2_two_level_interpolation = zeros(Nx_2,Ny_2)
interpolated_sol_2_one_level_interpolation = zeros(Nx_1,Ny_1)
matrix_free_prolongation_2d(sol_0,interpolated_sol_2_one_level_interpolation)

# using this initial guess for the CG
x_with_initial_guess_one_level, history_initial_guess_one_level = cg!(interpolated_sol_2_one_level_interpolation[:],A1,b1;abstol=norm(b2)*sqrt(eps(real(eltype(b2)))),log=true)
@show history_initial_guess_one_level.iters 
@show history_initial_guess_one_level.data[:resnorm]

# doing another interpolation using this result
matrix_free_prolongation_2d(reshape(x_with_initial_guess_one_level,Nx_1,Ny_1),interpolated_sol_2_two_level_interpolation)

x_with_initial_guess_two_level, history_initial_guess_two_level = cg!(interpolated_sol_2_two_level_interpolation[:],A2,b2;abstol=norm(b2)*sqrt(eps(real(eltype(b2)))),log=true)

@show history_initial_guess_two_level.iters 
@show history_initial_guess_two_level.data[:resnorm]