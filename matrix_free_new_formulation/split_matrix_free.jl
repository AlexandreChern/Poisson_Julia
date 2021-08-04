using SparseArrays 
using CUDA
using Random
using Adapt

function D2_cpu(idata,odata,Nx,Ny,h)
    for i in 1:Nx
        for j in 1:Ny
            if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
                odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
            end
        end
    end
end

function D2_split(idata,odata,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
        odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
    end 

    nothing
end



# function matrix_free_cpu_old(idata,odata,Nx,Ny,h)
#     (Nx,Ny) = size(idata)
#     # odata = spzeros(Nx,Ny)
#     idx = 1:3
#     odata[idx,:] .= idata[idx,:]
#     odata[Nx+1 .- idx,:] .= idata[Nx+1 .- idx,:]
#     odata
#     odata[:,idx] .= idata[:,idx]
#     odata[:,Ny+1 .- idx] .= idata[:,Ny+1 .- idx]
#     odata

#     alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
#     offset_j = spzeros(Int,Ny)
#     offset_j[2] = -1
#     offset_j[3] = -2
#     offset_j[Ny-1] = 1
#     offset_j[Ny-2] = 2
#     coef_j = spzeros(Ny)
#     coef_j[1] = 2* 1.5
#     coef_j[2] = - 2.0
#     coef_j[3] = 0.5
#     coef_j[Ny] = 2*1.5
#     coef_j[Ny-1] = -2.0
#     coef_j[Ny-2] = 0.5
#     for i in 1:Nx
#         for j = 1:Ny
#             if !(3<= i <= Nx-3) && (3 <= j <= Ny-3)
#                 global_index = (i-1)*Ny+j

#                 offset_x = div(2*Nx-i-3,Nx-2) - 1
#                 offset_y = div(2*Ny-j-3,Ny-2) - 1

#                 idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
#                 idata_index_y = (i-1)*Ny + j + offset_y

#                 odata[global_index] = (( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) )   
#                 + abs(offset_y) * 2 * alpha1 * ( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y])
#                 + abs(offset_x) * alpha4 * 2 * (idata[global_index] * h) 
#                 + coef_j[i] * beta * (idata[global_index+offset_j[i] * Ny])
#                 ) / 2^(abs(offset_x) + abs(offset_y))
#             end
#         end
#     end
# end

function matrix_free_cpu(idata,odata,Nx,Ny,h)
    # odata .= 0
    # for i in 1:Nx
    #     for j in 1:Ny
    #         if (i == 1) && (j == 1)
    #             odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    #         end
    #         if (i == 1) && (j == Ny)
    #             odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    #         end
    #         if (i == Nx) && (j == 1)
    #             odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    #         end
    #         # if (i == Nx) && (j == Ny)
    #         #     odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    #         # end
    #     end
    # end
    odata .= 0
    # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1
    (i,j) = (1,1)

    odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

    odata[i,j] += 2 * alpha3 * (( 1.5* idata[i,j] - 2*idata[i+1,j] + 0.5*idata[i+2,j])) / 4 # Neumann

    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


    (i,j) = (1,Ny)
    odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2
    
    odata[i,j] += 2 * alpha3 * (1.5 * idata[i,j] - 2*idata[i+1,j] + 0.5 * idata[i+2,j]) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet



    (i,j) = (Nx,1)
    odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

    odata[i,j] += 2 * alpha4 * (( 1.5* idata[i,j] - 2*idata[i-1,j] + 0.5*idata[i-2,j])) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet

    (i,j) = (Nx,Ny)
    odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2

    odata[i,j] += 2 * alpha4 * (1.5 * idata[i,j] - 2*idata[i-1,j] + 0.5 * idata[i-2,j]) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


    (i,j) = (1,2:Ny-1)
    # odata[i,j] .+= ((idata[i,j] .- 2*idata[i+1,j] .+ idata[i+2,j] .+ idata[i,j .- 1] .- 2*idata[i,j] .+ idata[i,j .+ 1] ) ./ 2 # D2
    #                 # .+ 2 .* alpha3 .* (1.5 * idata[i,j] .- 2*idata[i+1,j] .+ 0.5 * idata[i+2,j]) ./ 2
    #                 )
    # odata[i,j] .+= 2 * alpha3 * (1.5 * idata[i,j] .- 2*idata[i+1,j] .+ 0.5 * idata[i+2,j]) ./ 2 # Neumann

    # test @view
    odata[i,j] .+= (view(idata,i,j) .- 2*view(idata,i+1,j) .+ view(idata,i+2,j) .+ view(idata,i,j .-1) .- 2*view(idata,i,j) .+ view(idata,i,j .+ 1)
                    .+ 2 * alpha3 .* (1.5 * view(idata,i,j) .- 2*view(idata,i+1,j) .+ 0.5 * view(idata,i+2,j))) ./ 2

    (i,j) = (Nx,2:Ny-1)
    # odata[i,j] .+= (idata[i,j] .- 2*idata[i-1,j] .+ idata[i-2,j] .+ idata[i,j .- 1] .- 2*idata[i,j] .+ idata[i,j .+ 1]) / 2 # D2
    # odata[i,j] .+= 2 * alpha4 * (1.5 * idata[i,j] .- 2*idata[i-1,j] .+ 0.5 * idata[i-2,j]) ./ 2 # Neumann
    odata[i,j] .+= ((view(idata,i,j) .- 2*view(idata,i-1,j) .+ view(idata,i-2,j) .+ view(idata,i,j.-1) .- 2*view(idata,i,j) .+ view(idata,i,j.+1))
                    .+ 2 * alpha4 * (1.5 * view(idata,i,j) .- 2*view(idata,i-1,j) .+ 0.5 * view(idata,i-2,j))) ./ 2


    (i,j) = (2:Nx-1,1)
    # odata[i,j] .+= (idata[i.-1,j] .- 2*idata[i,j] .+ idata[i.+1,j] .+ idata[i,j] .- 2*idata[i,j+1] .+ idata[i,j+2]) / 2 # D2
    odata[i,j] .+= (view(idata,i .- 1,j) .- 2*view(idata,i,j) .+ view(idata,i .+ 1,j) .+ view(idata,i,j) .- 2*view(idata,i,j+1) .+ view(idata,i,j+2)) ./ 2

    # odata[i,j] .+= (2 * beta * (1.5 * idata[i,j]) .+ 2 * alpha1 * (idata[i,j]) * h) ./ 2 # Dirichlet
    # odata[i,j+1] .+= (2 * beta * (-1 * idata[i,j])) # Dirichlet
    # odata[i,j+2] .+= (0.5 * beta * (idata[i,j]))  # Dirichlet
    odata[i,j] .+= (2 * beta * (1.5 * view(idata,i,j)) .+ 2 * alpha2 * view(idata,i,j) * h) ./ 2
    odata[i,j+1] .+= (2 * beta * (-1 * view(idata,i,j)))
    odata[i,j+2] .+= (0.5 * beta * (view(idata,i,j)))


    (i,j) = (2:Nx-1,Ny)
    # odata[i,j] .+= (idata[i.-1,j] .- 2*idata[i,j] .+ idata[i.+1,j] .+ idata[i,j] .- 2*idata[i,j-1] .+ idata[i,j - 2]) / 2 # D2
    odata[i,j] .+= (view(idata,i .- 1, j) .- 2*view(idata,i,j) .+ view(idata,i .+1,j) .+ view(idata,i,j) .- 2*view(idata,i,j-1) .+ view(idata,i,j-2)) ./ 2

    # odata[i,j] .+= (2 * beta * (1.5 * idata[i,j]) .+ 2 * alpha1 * (idata[i,j]) * h) ./ 2 # Dirichlet
    # # odata[i,j-1] .+= (2 * beta * (-1 * idata[i,j])) # Dirichlet
    # # odata[i,j-2] .+= (0.5 * beta * (idata[i,j])) # Dirichlet
    odata[i,j] .+= (2 * beta * (1.5 * view(idata,i,j)) .+ 2 * alpha1 * view(idata,i,j) * h) ./ 2
    odata[i,j-1] .+= (2 * beta * (-1 * view(idata,i,j)))
    odata[i,j-2] .+= (0.5 * beta * (view(idata,i,j)))
end

function matrix_free_cpu_v2(idata,odata,Nx,Ny,h)
    # odata .= 0
    # for i in 1:Nx
    #     for j in 1:Ny
    #         if (i == 1) && (j == 1)
    #             odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    #         end
    #         if (i == 1) && (j == Ny)
    #             odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    #         end
    #         if (i == Nx) && (j == 1)
    #             odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    #         end
    #         # if (i == Nx) && (j == Ny)
    #         #     odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    #         # end
    #     end
    # end
    odata .= 0
    # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1
    (i,j) = (1,1)

    odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

    odata[i,j] += 2 * alpha3 * (( 1.5* idata[i,j] - 2*idata[i+1,j] + 0.5*idata[i+2,j])) / 4 # Neumann

    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


    (i,j) = (1,Ny)
    odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2
    
    odata[i,j] += 2 * alpha3 * (1.5 * idata[i,j] - 2*idata[i+1,j] + 0.5 * idata[i+2,j]) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet



    (i,j) = (Nx,1)
    odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

    odata[i,j] += 2 * alpha4 * (( 1.5* idata[i,j] - 2*idata[i-1,j] + 0.5*idata[i-2,j])) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet

    (i,j) = (Nx,Ny)
    odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2

    odata[i,j] += 2 * alpha4 * (1.5 * idata[i,j] - 2*idata[i-1,j] + 0.5 * idata[i-2,j]) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


    (i,j) = (1,2:Ny-1)
    odata[i,j] .+= (view(idata,i,j) .- 2*view(idata,i+1,j) .+ view(idata,i+2,j) .+ view(idata,i,j .-1) .- 2*view(idata,i,j) .+ view(idata,i,j .+ 1)
                    .+ 2 * alpha3 .* (1.5 * view(idata,i,j) .- 2*view(idata,i+1,j) .+ 0.5 * view(idata,i+2,j))) ./ 2

    (i,j) = (Nx,2:Ny-1)
    odata[i,j] .+= ((view(idata,i,j) .- 2*view(idata,i-1,j) .+ view(idata,i-2,j) .+ view(idata,i,j.-1) .- 2*view(idata,i,j) .+ view(idata,i,j.+1))
                    .+ 2 * alpha4 * (1.5 * view(idata,i,j) .- 2*view(idata,i-1,j) .+ 0.5 * view(idata,i-2,j))) ./ 2


    (i,j) = (2:Nx-1,1)
    odata[i,j] .+= (view(idata,i .- 1,j) .- 2*view(idata,i,j) .+ view(idata,i .+ 1,j) .+ view(idata,i,j) .- 2*view(idata,i,j+1) .+ view(idata,i,j+2)) ./ 2

    odata[i,j] .+= (2 * beta * (1.5 * view(idata,i,j)) .+ 2 * alpha2 * view(idata,i,j) * h) ./ 2
    odata[i,j+1] .+= (2 * beta * (-1 * view(idata,i,j)))
    odata[i,j+2] .+= (0.5 * beta * (view(idata,i,j)))


    (i,j) = (2:Nx-1,Ny)
    odata[i,j] .+= (view(idata,i .- 1, j) .- 2*view(idata,i,j) .+ view(idata,i .+1,j) .+ view(idata,i,j) .- 2*view(idata,i,j-1) .+ view(idata,i,j-2)) ./ 2

    odata[i,j] .+= (2 * beta * (1.5 * view(idata,i,j)) .+ 2 * alpha1 * view(idata,i,j) * h) ./ 2
    odata[i,j-1] .+= (2 * beta * (-1 * view(idata,i,j)))
    odata[i,j-2] .+= (0.5 * beta * (view(idata,i,j)))
end


function matrix_free_cpu_v3(idata,odata,Nx,Ny,h)
    # odata .= 0
    # for i in 1:Nx
    #     for j in 1:Ny
    #         if (i == 1) && (j == 1)
    #             odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    #         end
    #         if (i == 1) && (j == Ny)
    #             odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    #         end
    #         if (i == Nx) && (j == 1)
    #             odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    #         end
    #         # if (i == Nx) && (j == Ny)
    #         #     odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    #         # end
    #     end
    # end
    odata .= 0
    # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1
    (i,j) = (1,1)

    odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

    odata[i,j] += 2 * alpha3 * (( 1.5* idata[i,j] - 2*idata[i+1,j] + 0.5*idata[i+2,j])) / 4 # Neumann

    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


    (i,j) = (1,Ny)
    odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2
    
    odata[i,j] += 2 * alpha3 * (1.5 * idata[i,j] - 2*idata[i+1,j] + 0.5 * idata[i+2,j]) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet



    (i,j) = (Nx,1)
    odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

    odata[i,j] += 2 * alpha4 * (( 1.5* idata[i,j] - 2*idata[i-1,j] + 0.5*idata[i-2,j])) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet

    (i,j) = (Nx,Ny)
    odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2

    odata[i,j] += 2 * alpha4 * (1.5 * idata[i,j] - 2*idata[i-1,j] + 0.5 * idata[i-2,j]) / 4 # Neumann
    odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
    odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
    odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


    # (i,j) = (1,2:Ny-1)
    i = 1
    idata_N = view(idata,1:3,1:Ny)
    # Threads.@threads for j in 2:Ny-1
    @inbounds for j in 2:Ny-1
        odata[1,j] += (idata_N[1,j] - 2*idata_N[2,j] + idata_N[3,j] + idata_N[1,j-1] - 2* idata_N[1,j] + idata_N[1,j+1] + 2 * alpha3 * (1.5 * idata_N[1,j] - 2*idata_N[2,j] + 0.5*idata_N[3,j])) / 2
    end
    # synchronize()

    i = Nx
    idata_S = view(idata,Nx-2:Nx,1:Ny)
    # Threads.@threads for j in 2:Ny-1
    @inbounds for j in 2:Ny-1
        odata[i,j] += (idata_S[3,j] - 2*idata_S[2,j] + idata_S[1,j] + idata_S[3,j-1] - 2* idata_S[3,j] + idata_S[3,j+1] + 2 * alpha4 * (1.5 * idata_S[3,j] - 2*idata_S[2,j] + 0.5*idata_S[1,j])) / 2
    end
    # synchronize()

    j = 1
    idata_W = view(idata,1:Nx,1:3)

    # @inbounds for i in 2:Nx-1
    #     odata[i,j] += (idata_W[i-1,1] - 2*idata_W[i,1] + idata_W[i+1,1] + idata_W[i,1] - 2*idata_W[i,2] + idata_W[i,3]) / 2
    #     # odata[i,j] += (2 * beta * (1.5 * idata_W[i,1]) + 2 * alpha2 * idata_W[i,1] * h) / 2
    #     # odata[i,j+1] += (2 * beta * (-1 * idata_W[i,1]))
    #     # odata[i,j+2] += (0.5 * beta * idata_W[i,1])
    # end

    idata_W_T = copy(idata_W')
    odata_W_T= zeros(size(idata_W_T))
    # @inbounds for i in 2:Nx-1
    @inbounds for i in 2:Nx-1
        odata_W_T[1,i] += (idata_W_T[1,i-1] - 2*idata_W_T[1,i] + idata_W_T[1,i+1] + idata_W_T[1,i] - 2*idata_W_T[2,i] + idata_W_T[3,i]) / 2
        odata_W_T[1,i] += (2 * beta * (1.5 * idata_W_T[1,i]) + 2 * alpha2 * idata_W_T[1,i] * h) / 2
        odata_W_T[2,i] += (2 * beta * (-1 * idata_W_T[1,i]))
        odata_W_T[3,i] += (0.5 * beta * idata_W_T[1,i])
    end

    # odata[:,1] .+= odata_W_T[1,:]

   

    j = Ny
    idata_E = view(idata,1:Nx,Ny-2:Ny)
    idata_E_T = copy(idata_E')
    odata_E_T = zeros(size(idata_E_T))
    @inbounds for i in 2:Nx-1
        odata_E_T[3,i] += (idata_E_T[3,i-1] - 2*idata_E_T[3,i] + idata_E_T[3,i+1] + idata_E_T[3,i] - 2*idata_E_T[2,i] + idata_E_T[1,i]) / 2
        odata_E_T[3,i] += (2 * beta * (1.5 * idata_E_T[3,i]) + 2 * alpha1 * idata_E_T[3,i] * h) / 2
        odata_E_T[2,i] += (2 * beta * (-1 * idata_E_T[3,i]))
        odata_E_T[1,i] += (0.5 * beta * idata_E_T[3,i])
    end

    # @inbounds for i in 2:Nx-1
    #     odata[i,j] += (idata_E[i-1,3] - 2*idata_E[i,3] + idata_E[i+1,3] + idata_E[i,3] - 2*idata_E[i,2] + idata_E[i,1]) / 2
    #     odata[i,j] += (2 * beta * (1.5 * idata_E[i,3]) + 2 * alpha1 * idata_E[i,3] * h) / 2
    #     odata[i,j-1] += (2 * beta * (-1 * idata_E[i,3]))
    #     odata[i,j-2] += (0.5 * beta * idata_E[i,3])
    # end

    # odata[:,1:3] .+= odata_W_T'
    # odata[:,end-2:end] .+= odata_E_T'
end

function test_matrix_free_boundary(level)
    Nx = Ny = 2^level+1
    h = 1/(Nx-1)
    Random.seed!(0)
    A = randn(Nx,Ny)
    A_sparse = spzeros(Nx,Ny)
    for i in 1:Nx
        for j in 1:Ny
            if !(( 4 <= i <= Nx - 4) && (4 <= j <= Ny-4))
                A_sparse[i,j] = A[i,j]
            end
        end
    end
    odata = spzeros(Nx,Ny)
    odata_v3 = spzeros(Nx,Ny)

    # # make sure matrix_free_cpu_v3 equals matrix_free_cpu
    # odata_test = spzeros(Nx,Ny)
    # matrix_free_cpu_v3(A,odata_test,Nx,Ny,h)
    # matrix_free_cpu(A,odata,Nx,Ny,h)
    # # @show Array(odata)
    # # @show Array(odata_test)
    # @assert odata ≈ odata_test
    # # end test

    # t_cpu = time()
    # iter_times = 1000
    # for i in 1:iter_times
    #     matrix_free_cpu_v2(A_sparse,odata,Nx,Ny,h)
    #     odata .= 0
    #     # matrix_free_cpu_v3(A,odata,Nx,Ny,h)
    # end
    # t_cpu = time() - t_cpu
    # @show t_cpu

    t_cpu_v3 = time()
    iter_times = 1000
    for i in 1:iter_times
        matrix_free_cpu_v3(A,odata_v3,Nx,Ny,h)
        odata_v3 .= 0
        # matrix_free_cpu_v3(A,odata,Nx,Ny,h)
    end
    synchronize()
    t_cpu_v3 = time() - t_cpu_v3
    @show t_cpu_v3

    t_convert = time()
    for i in 1:iter_times
        cu_out = CUDA.CUSPARSE.CuSparseMatrixCSC(odata)
    end
    synchronize()
    t_convert = time() - t_convert
    @show t_convert
    nothing
end


function test_D2_split(level)
    Nx = Ny = 2^level+1
    h = 1/(Nx-1)
    Random.seed!(0)
    A = randn(Nx,Ny)
    A = sparse(A)
    # cu_A = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    # cu_A = CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    # cu_out = CUDA.CUSPARSE.CuSparseMatrixCSC(spzeros(Nx,Ny))
    cu_A = CuArray(A)
    cu_out = CuArray(zeros(Nx,Ny))
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

    @cuda threads=blockdim blocks=griddim D2_split(cu_A,cu_out,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))

    iter_times = 1000

    t_D2_start = time()
    for _ in 1:iter_times
        @cuda threads=blockdim blocks=griddim D2_split(cu_A,cu_out,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    end
    synchronize()
    t_D2 = time() - t_D2_start

    @show t_D2
    nothing
end



function matrix_free_A(idata,odata)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    odata_cpu = spzeros(Nx,Ny)
    # odata_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(spzeros(Nx,Ny))
    odata_gpu = CuArray(zeros(Nx,Ny))
    # idata_gpu = CUDA.CUSPARSE.CuSparseMatrixCSC(idata)
    idata_gpu = CUDA.CuArray(idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim D2_split(idata_gpu,odata_gpu,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    matrix_free_cpu_v2(idata,odata_cpu,Nx,Ny,h)
    odata .= adapt(Array,odata_gpu) .+ odata_cpu
end

function matrix_free_A_v2(idata,odata)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    # odata_gpu = CuArray(zeros(Nx,Ny))
    # idata_gpu = CUDA.CuArray(idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim D2_split(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
end

function test_matrix_free_A(level)
    Nx = Ny = 2^level + 1
    h = 1/(Nx-1)
    println("2D Domain Size: $Nx by $Ny")
    Random.seed!(0)
    # idata = sparse(randn(Nx,Ny))
    idata = CuArray(randn(Nx,Ny))
    # odata = spzeros(Nx,Ny)
    odata = CuArray(randn(Nx,Ny))
    println("Size of the solution matrix (GPU): ", sizeof(idata), " Bytes")

    idata_cpu = zeros(Nx,Ny)
    odata_cpu = spzeros(Nx,Ny)
    copyto!(idata_cpu,idata)
    println("Size of the solution matrix (CPU): ", sizeof(idata_cpu), " Bytes")

    println("")

    println("Evaluting time to do one A * b")
    println("Timing results in ms")
    println("")

    iter_times = 1000

    #precompile functions
    matrix_free_A_v2(idata,odata)
    matrix_free_cpu_v3(idata_cpu,odata_cpu,Nx,Ny,h)



    # Evaluating only D2
    t_start_D2 = time()
    for _ in 1:iter_times
        matrix_free_A_v2(idata,odata)
    end
    synchronize()
    t_D2 = (time() - t_start_D2) * 1000 / iter_times
    @show t_D2
    # End evaluating D2


    # Evaluating only boundary
    t_start_boundary = time()
    for _ in 1:iter_times
        matrix_free_cpu_v3(idata_cpu,odata_cpu,Nx,Ny,h)
    end
    t_boundary = (time() - t_start_boundary) * 1000 / iter_times
    @show t_boundary
    # End evaluating boundary


    # Evaluating Both in asynchronous way
    t_start_total = time()
    for _ in 1:iter_times
        matrix_free_A_v2(idata,odata)
    end
    
    for _ in 1:iter_times
        matrix_free_cpu_v3(idata_cpu,odata_cpu,Nx,Ny,h)
    end
    synchronize()
    t_total = (time() - t_start_total) * 1000 / iter_times
    @show t_total 
    # End evaluating both in asynchrnous way

    # Evaluating time in Data IO
    t_copy_data = time()
    iter_times_copy_data = 20
    for _ in 1:iter_times_copy_data
        copyto!(idata_cpu,idata)
    end
    # End evaluating time in Data IO
    t_copy_data = ( time() - t_copy_data ) * 1000 / iter_times_copy_data
    @show t_copy_data 
    nothing
end



test_matrix_free_A(13)
test_matrix_free_A(14)