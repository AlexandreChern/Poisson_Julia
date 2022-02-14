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

function D2_split_dev(idata,odata,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    if 0 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = 0
    end

    if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
        odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
    end 

    nothing
end




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

    CPU_W_T = copy(idata_W')
    odata_W_T= zeros(size(CPU_W_T))
    # @inbounds for i in 2:Nx-1
    @inbounds for i in 2:Nx-1
        odata_W_T[1,i] += (CPU_W_T[1,i-1] - 2*CPU_W_T[1,i] + CPU_W_T[1,i+1] + CPU_W_T[1,i] - 2*CPU_W_T[2,i] + CPU_W_T[3,i]) / 2
        odata_W_T[1,i] += (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        odata_W_T[2,i] += (2 * beta * (-1 * CPU_W_T[1,i]))
        odata_W_T[3,i] += (0.5 * beta * CPU_W_T[1,i])
    end

    # odata[:,1] .+= odata_W_T[1,:]

   

    j = Ny
    idata_E = view(idata,1:Nx,Ny-2:Ny)
    CPU_E_T = copy(idata_E')
    CPU_OUT_E_T = zeros(size(CPU_E_T))
    @inbounds for i in 2:Nx-1
        CPU_OUT_E_T[3,i] += (CPU_E_T[3,i-1] - 2*CPU_E_T[3,i] + CPU_E_T[3,i+1] + CPU_E_T[3,i] - 2*CPU_E_T[2,i] + CPU_E_T[1,i]) / 2
        CPU_OUT_E_T[3,i] += (2 * beta * (1.5 * CPU_E_T[3,i]) + 2 * alpha1 * CPU_E_T[3,i] * h) / 2
        CPU_OUT_E_T[2,i] += (2 * beta * (-1 * CPU_E_T[3,i]))
        CPU_OUT_E_T[1,i] += (0.5 * beta * CPU_E_T[3,i])
    end

    # @inbounds for i in 2:Nx-1
    #     odata[i,j] += (idata_E[i-1,3] - 2*idata_E[i,3] + idata_E[i+1,3] + idata_E[i,3] - 2*idata_E[i,2] + idata_E[i,1]) / 2
    #     odata[i,j] += (2 * beta * (1.5 * idata_E[i,3]) + 2 * alpha1 * idata_E[i,3] * h) / 2
    #     odata[i,j-1] += (2 * beta * (-1 * idata_E[i,3]))
    #     odata[i,j-2] += (0.5 * beta * idata_E[i,3])
    # end

    odata[:,1:3] .+= odata_W_T'
    odata[:,end-2:end] .+= CPU_OUT_E_T'
    nothing
end

function matrix_free_cpu_optimized(GPU_Array,odata,Nx,Ny,h)
    # input as idata_gpu
    # Calculation using CPU

    odata .= 0

    CPU_W = zeros(Nx,3)
    CPU_W_T = zeros(3,Nx)
    CPU_E = zeros(Nx,3)
    CPU_E_T = zeros(3,Nx)
    CPU_N = zeros(3,Ny)
    CPU_S = zeros(3,Ny)


    CPU_OUT_W = zeros(Nx,3)
    CPU_OUT_E = zeros(Nx,3)
    CPU_OUT_N = zeros(3,Ny)
    CPU_OUT_S = zeros(3,Ny)

    copyto!(CPU_W,GPU_Array[:,1:3])
    copyto!(CPU_E,GPU_Array[:,end-2:end])
    copyto!(CPU_N,GPU_Array[1:3,:])
    copyto!(CPU_S,GPU_Array[end-2:end,:])

    # copyto!(CPU_W,view(GPU_Array,1:Nx,1:3))
    # copyto!(CPU_E,view(GPU_Array,1:Nx,Ny-2:Ny))
    # copyto!(CPU_N,view(GPU_Array,1:3,:))
    # copyto!(CPU_S,view(GPU_Array,Nx-2:Nx,:))

    CPU_W_T .= CPU_W'
    CPU_E_T .= CPU_E'
    CPU_OUT_W_T = zeros(3,Ny)
    CPU_OUT_E_T = zeros(3,Ny)


    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1

    # (i,j) = (1,2:Ny-1)
    i = 1
    # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        CPU_OUT_N[1,j] += (CPU_N[1,j] - 2*CPU_N[2,j] + CPU_N[3,j] + CPU_N[1,j-1] - 2* CPU_N[1,j] + CPU_N[1,j+1] + 2 * alpha3 * (1.5 * CPU_N[1,j] - 2*CPU_N[2,j] + 0.5*CPU_N[3,j])) / 2
    end
    synchronize()

    i = Nx
    # # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        CPU_OUT_S[3,j] += (CPU_S[3,j] - 2*CPU_S[2,j] + CPU_S[1,j] + CPU_S[3,j-1] - 2* CPU_S[3,j] + CPU_S[3,j+1] + 2 * alpha4 * (1.5 * CPU_S[3,j] - 2*CPU_S[2,j] + 0.5*CPU_S[1,j])) / 2
    end
    # synchronize()

    j = 1

    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        CPU_OUT_W_T[1,i] += (CPU_W_T[1,i-1] - 2*CPU_W_T[1,i] + CPU_W_T[1,i+1] + CPU_W_T[1,i] - 2*CPU_W_T[2,i] + CPU_W_T[3,i]) / 2
        CPU_OUT_W_T[1,i] += (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        CPU_OUT_W_T[2,i] += (2 * beta * (-1 * CPU_W_T[1,i]))
        CPU_OUT_W_T[3,i] += (0.5 * beta * CPU_W_T[1,i])
    end

    # # odata[:,1] .+= odata_W_T[1,:]

   

    j = Ny
    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        CPU_OUT_E_T[3,i] += (CPU_E_T[3,i-1] - 2*CPU_E_T[3,i] + CPU_E_T[3,i+1] + CPU_E_T[3,i] - 2*CPU_E_T[2,i] + CPU_E_T[1,i]) / 2
        CPU_OUT_E_T[3,i] += (2 * beta * (1.5 * CPU_E_T[3,i]) + 2 * alpha1 * CPU_E_T[3,i] * h) / 2
        CPU_OUT_E_T[2,i] += (2 * beta * (-1 * CPU_E_T[3,i]))
        CPU_OUT_E_T[1,i] += (0.5 * beta * CPU_E_T[3,i])
    end

    (i,j) = (1,1)


    CPU_OUT_N[1,j] += (CPU_W[i,j] - 2*CPU_W[i+1,j] + CPU_W[i+2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    CPU_OUT_N[1,j] += 2 * alpha3 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i+1,j] + 0.5*CPU_W[i+2,j])) / 4 # Neumann

    CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    CPU_OUT_N[1,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    CPU_OUT_N[1,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet


    (i,j) = (1,Ny)
    CPU_OUT_N[1,j] += (CPU_E[i,3] - 2*CPU_E[i+1,3] + CPU_E[i+2,3] + CPU_E[i,3] - 2*CPU_E[i,2] + CPU_E[i,1]) / 4 # D2
    
    CPU_OUT_N[1,j] += 2 * alpha3 * (1.5 * CPU_E[i,3] - 2*CPU_E[i+1,3] + 0.5 * CPU_E[i+2,3]) / 4 # Neumann
    CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_E[i,3]) + 2 * alpha2 * (CPU_E[i,3]) * h) / 4 # Dirichlet
    CPU_OUT_N[1,j-1] += (2 * beta * (-1 * CPU_E[i,3])) / 2 # Dirichlet
    CPU_OUT_N[1,j-2] += (0.5 * beta * (CPU_E[i,3])) / 2 # Dirichlet



    (i,j) = (Nx,1)
    CPU_OUT_S[3,j] += (CPU_W[i,j] - 2*CPU_W[i-1,j] + CPU_W[i-2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    CPU_OUT_S[3,j] += 2 * alpha4 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i-1,j] + 0.5*CPU_W[i-2,j])) / 4 # Neumann
    CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    CPU_OUT_S[3,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    CPU_OUT_S[3,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet

    (i,j) = (Nx,Ny)
    CPU_OUT_S[3,j] += (CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + CPU_E[Nx-2,3] + CPU_E[Nx,3] - 2*CPU_E[Nx,2] + CPU_E[Nx,1]) / 4 # D2

    CPU_OUT_S[3,j] += 2 * alpha4 * (1.5 * CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + 0.5 * CPU_E[Nx-2,3]) / 4 # Neumann
    CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_E[Nx,3]) + 2 * alpha2 * (CPU_E[Nx,3]) * h) / 4 # Dirichlet
    CPU_OUT_S[3,j-1] += (2 * beta * (-1 * CPU_E[Nx,3])) / 2 # Dirichlet
    CPU_OUT_S[3,j-2] += (0.5 * beta * (CPU_E[Nx,3])) / 2 # Dirichlet


    # CPU_OUT_cache = CuArray(CPU_OUT_W_T)'
    # CPU_OUT_cache = CuArray(CPU_OUT_E_T)'
    # CuArray(CPU_OUT_W_T')
    # CuArray(CPU_OUT_E_T')
    # @show CPU_OUT_W_T
    # @show CPU_OUT_E_T
    # CPU_OUT_W .= CPU_OUT_W_T'
 
    # Copy E & W boundary
    copyto!(view(odata,1:Nx,1:3),CuArray(CPU_OUT_W_T)')
    copyto!(view(odata,1:Nx,Ny-2:Ny),CuArray(CPU_OUT_E_T)')

    # Copy N & S boundary
    copyto!(view(odata,1:1,1:Ny),CuArray(CPU_OUT_N[1,:]))
    copyto!(view(odata,Nx:Nx,1:Ny),CuArray(CPU_OUT_S[end,:]))
    nothing
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

    @cuda threads=blockdim blocks=griddim D2_split_dev(cu_A,cu_out,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))

    iter_times = 1000

    t_D2_start = time()
    for _ in 1:iter_times
        @cuda threads=blockdim blocks=griddim D2_split_dev(cu_A,cu_out,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    end
    synchronize()
    t_D2 = time() - t_D2_start

    @show t_D2
    nothing
end



function matrix_free_A_v1(idata,odata)
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
    @cuda threads=blockdim blocks=griddim D2_split_dev(idata_gpu,odata_gpu,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
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
    @cuda threads=blockdim blocks=griddim D2_split_dev(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
end

function matrix_free_A_v3(idata,odata,odata_D2_GPU,odata_boundary_GPU)
    # final version for CG
    # odata .= 0
    Nx,Ny = size(idata)
    # odata_D2 = CUDA.zeros(Nx,Ny)
    # odata_boundary = CUDA.zeros(Nx,Ny)
    # odata_boundary = CuArray(zeros(Nx,Ny))
    h = 1/(Nx-1)
    # odata_gpu = CuArray(zeros(Nx,Ny))
    # idata_gpu = CUDA.CuArray(idata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim D2_split_dev(idata,odata_D2_GPU,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    matrix_free_cpu_optimized(idata,odata_boundary_GPU,Nx,Ny,h)
    synchronize()
    odata .= odata_D2_GPU .+ odata_boundary_GPU
    # copyto!(view(odata,1:3,:), view(odata_D2_GPU,1:3,:) + view(odata_boundary_GPU,1:3,:))
    # return odata
end


function matrix_free_A_v4(idata,odata)
    # odata .= 0
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    # odata = CUDA.zeros(Nx,Ny)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

    CPU_W = zeros(Nx,3)
    CPU_W_T = zeros(3,Nx)
    CPU_E = zeros(Nx,3)
    CPU_E_T = zeros(3,Nx)
    CPU_N = zeros(3,Ny)
    CPU_S = zeros(3,Ny)


    CPU_OUT_W = zeros(Nx,3)
    CPU_OUT_E = zeros(Nx,3)
    CPU_OUT_N = zeros(3,Ny)
    CPU_OUT_S = zeros(3,Ny)

    copyto!(CPU_W,idata[:,1:3])
    copyto!(CPU_E,idata[:,end-2:end])
    copyto!(CPU_N,idata[1:3,:])
    copyto!(CPU_S,idata[end-2:end,:])
    # copyto!(CPU_W,GPU_Array[:,1:3])
    # copyto!(CPU_E,GPU_Array[:,end-2:end])
    # copyto!(CPU_N,GPU_Array[1:3,:])
    # copyto!(CPU_S,GPU_Array[end-2:end,:])

    @cuda threads=blockdim blocks=griddim D2_split_dev(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))

    CPU_W_T .= CPU_W'
    CPU_E_T .= CPU_E'
    CPU_OUT_W_T = zeros(3,Ny)
    CPU_OUT_E_T = zeros(3,Ny)


    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1

    i = 1
    # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        CPU_OUT_N[1,j] += (CPU_N[1,j] - 2*CPU_N[2,j] + CPU_N[3,j] + CPU_N[1,j-1] - 2* CPU_N[1,j] + CPU_N[1,j+1] + 2 * alpha3 * (1.5 * CPU_N[1,j] - 2*CPU_N[2,j] + 0.5*CPU_N[3,j])) / 2
    end
    synchronize()

    i = Nx
    # # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        CPU_OUT_S[3,j] += (CPU_S[3,j] - 2*CPU_S[2,j] + CPU_S[1,j] + CPU_S[3,j-1] - 2* CPU_S[3,j] + CPU_S[3,j+1] + 2 * alpha4 * (1.5 * CPU_S[3,j] - 2*CPU_S[2,j] + 0.5*CPU_S[1,j])) / 2
    end
    # synchronize()

    j = 1

    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        CPU_OUT_W_T[1,i] += (CPU_W_T[1,i-1] - 2*CPU_W_T[1,i] + CPU_W_T[1,i+1] + CPU_W_T[1,i] - 2*CPU_W_T[2,i] + CPU_W_T[3,i]) / 2
        CPU_OUT_W_T[1,i] += (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        CPU_OUT_W_T[2,i] += (2 * beta * (-1 * CPU_W_T[1,i]))
        CPU_OUT_W_T[3,i] += (0.5 * beta * CPU_W_T[1,i])
    end

    # # odata[:,1] .+= odata_W_T[1,:]

   

    j = Ny
    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        CPU_OUT_E_T[3,i] += (CPU_E_T[3,i-1] - 2*CPU_E_T[3,i] + CPU_E_T[3,i+1] + CPU_E_T[3,i] - 2*CPU_E_T[2,i] + CPU_E_T[1,i]) / 2
        CPU_OUT_E_T[3,i] += (2 * beta * (1.5 * CPU_E_T[3,i]) + 2 * alpha1 * CPU_E_T[3,i] * h) / 2
        CPU_OUT_E_T[2,i] += (2 * beta * (-1 * CPU_E_T[3,i]))
        CPU_OUT_E_T[1,i] += (0.5 * beta * CPU_E_T[3,i])
    end

    (i,j) = (1,1)


    CPU_OUT_N[1,j] += (CPU_W[i,j] - 2*CPU_W[i+1,j] + CPU_W[i+2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    CPU_OUT_N[1,j] += 2 * alpha3 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i+1,j] + 0.5*CPU_W[i+2,j])) / 4 # Neumann

    CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    CPU_OUT_N[1,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    CPU_OUT_N[1,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet


    (i,j) = (1,Ny)
    CPU_OUT_N[1,j] += (CPU_E[i,3] - 2*CPU_E[i+1,3] + CPU_E[i+2,3] + CPU_E[i,3] - 2*CPU_E[i,2] + CPU_E[i,1]) / 4 # D2
    
    CPU_OUT_N[1,j] += 2 * alpha3 * (1.5 * CPU_E[i,3] - 2*CPU_E[i+1,3] + 0.5 * CPU_E[i+2,3]) / 4 # Neumann
    CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_E[i,3]) + 2 * alpha2 * (CPU_E[i,3]) * h) / 4 # Dirichlet
    CPU_OUT_N[1,j-1] += (2 * beta * (-1 * CPU_E[i,3])) / 2 # Dirichlet
    CPU_OUT_N[1,j-2] += (0.5 * beta * (CPU_E[i,3])) / 2 # Dirichlet



    (i,j) = (Nx,1)
    CPU_OUT_S[3,j] += (CPU_W[i,j] - 2*CPU_W[i-1,j] + CPU_W[i-2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    CPU_OUT_S[3,j] += 2 * alpha4 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i-1,j] + 0.5*CPU_W[i-2,j])) / 4 # Neumann
    CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    CPU_OUT_S[3,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    CPU_OUT_S[3,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet

    (i,j) = (Nx,Ny)
    CPU_OUT_S[3,j] += (CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + CPU_E[Nx-2,3] + CPU_E[Nx,3] - 2*CPU_E[Nx,2] + CPU_E[Nx,1]) / 4 # D2

    CPU_OUT_S[3,j] += 2 * alpha4 * (1.5 * CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + 0.5 * CPU_E[Nx-2,3]) / 4 # Neumann
    CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_E[Nx,3]) + 2 * alpha2 * (CPU_E[Nx,3]) * h) / 4 # Dirichlet
    CPU_OUT_S[3,j-1] += (2 * beta * (-1 * CPU_E[Nx,3])) / 2 # Dirichlet
    CPU_OUT_S[3,j-2] += (0.5 * beta * (CPU_E[Nx,3])) / 2 # Dirichlet

    synchronize()

    # Copy W & E boundary
    copyto!(view(odata,1:Nx,1:3),view(odata,1:Nx,1:3) + CuArray(CPU_OUT_W_T)')
    copyto!(view(odata,1:Nx,Ny-2:Ny),view(odata,1:Nx,Ny-2:Ny) + CuArray(CPU_OUT_E_T)')

    # view(odata,1,1:Ny) + CuArray(CPU_OUT_N[1,:])
    # view(odata,Nx,1:Ny) + CuArray(CPU_OUT_S[end,:])
    # # @show size(view(odata,1:1,1:Ny))
    # # @show size(CuArray(CPU_OUT_N[1,:]))


    # Copy N & S boundary
    copyto!(view(odata,1,1:Ny),view(odata,1,1:Ny) + CuArray(CPU_OUT_N[1,:]))
    copyto!(view(odata,Nx,1:Ny),view(odata,Nx,1:Ny) + CuArray(CPU_OUT_S[end,:]))
    nothing
end

function CG_GPU_dev(b_reshaped_GPU,x_GPU)
    (Nx,Ny) = size(b_reshaped_GPU)
    odata = CUDA.zeros(Nx,Ny)
    # odata_D2_GPU = CUDA.zeros(Nx,Ny)
    # odata_boundary_GPU = CUDA.zeros(Nx,Ny)
    # matrix_free_A_v3(x_GPU,odata,odata_D2_GPU,odata_boundary_GPU)
    matrix_free_A_v4(x_GPU,odata)
    r_GPU = b_reshaped_GPU - odata
    p_GPU = copy(r_GPU)
    rsold_GPU = sum(r_GPU .* r_GPU)
    Ap_GPU = CUDA.zeros(Nx,Ny)
    # num_steps = 0
    # @show rsold_GPU
    for i in 1:Nx*Ny
    # for i in 1:20
        # @show i
        # @show rsold_GPU
        # matrix_free_A_v3(p_GPU,Ap_GPU,odata_D2_GPU,odata_boundary_GPU)
        # num_steps += 1
        matrix_free_A_v4(p_GPU,Ap_GPU)
        alpha_GPU = rsold_GPU / (sum(p_GPU .* Ap_GPU))
        x_GPU .= x_GPU .+ alpha_GPU * p_GPU
        r_GPU .= r_GPU .- alpha_GPU * Ap_GPU
        rsnew_GPU = sum(r_GPU .* r_GPU)
        if sqrt(rsnew_GPU) <  sqrt(eps(real(eltype(b_reshaped_GPU))))
            break
        end
        p_GPU .= r_GPU .+ (rsnew_GPU/rsold_GPU) * p_GPU
        rsold_GPU = rsnew_GPU
        # @show rsold_GPU
    end
    # @show num_steps
end

function CG_CPU_dev(A,b,x)
    r = b - A * x;
    p = r;
    rsold = r' * r
    # Ap = p
    for i = 1:length(b)
    # for i = 1:2
        # @show i
        # @show rsold
        Ap = A * p;
        alpha = rsold / (p' * Ap)
        x .= x .+ alpha * p;
        r .= r .- alpha * Ap;
        rsnew = r' * r
        if sqrt(rsnew) < sqrt(eps(real(eltype(b))))
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew
        # @show rsold
    end
end


function test_CG()
    x = zeros(Nx*Ny)
    x_GPU = CUDA.zeros(Nx,Ny)
    CG_GPU(b_reshaped_GPU,x_GPU)
    CG_CPU(A,b,x)
end


function test_matrix_free_A(level)
    Nx = Ny = 2^level + 1
    h = 1/(Nx-1)
    println("")
    println("Starting Test")
    println("2D Domain Size: $Nx by $Ny")
    Random.seed!(0)
    # idata = sparse(randn(Nx,Ny))
    idata = CuArray(randn(Nx,Ny))
    # odata = spzeros(Nx,Ny)
    odata = CUDA.zeros(Nx,Ny)
    # odata_v4 = CUDA.zeros(Nx,Ny)
    # odata_D2_GPU = CUDA.zeros(Nx,Ny)
    odata_boundary_GPU = CUDA.zeros(Nx,Ny)
    println("Size of the solution matrix (GPU): ", sizeof(idata), " Bytes")

    idata_cpu = zeros(Nx,Ny)
    odata_cpu = spzeros(Nx,Ny)
    odata_D2 = zeros(Nx,Ny)
    copyto!(idata_cpu,idata)
    println("Size of the solution matrix (CPU): ", sizeof(idata_cpu), " Bytes")
    matrix_free_cpu(idata_cpu,odata_cpu,Nx,Ny,h)
    D2_cpu(idata_cpu,odata_D2,Nx,Ny,h)
    odata_cpu + odata_D2


    println("")

    println("Evaluting time to do one A * b")
    println("Timing results in ms")
    println("")

    iter_times = 1000

    #precompile functions

   
    # odata_boundary = CUDA.zeros(Nx,Ny)

    # matrix_free_A_v3(idata,odata,odata_D2_GPU,odata_boundary_GPU)
    matrix_free_A_v4(idata,odata)
    matrix_free_cpu_v3(idata_cpu,odata_cpu,Nx,Ny,h)
    matrix_free_cpu_optimized(idata,odata_boundary_GPU,Nx,Ny,h)

    @assert odata_cpu ≈ Array(odata_boundary_GPU)



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
        # matrix_free_cpu_v3(idata_cpu,odata_cpu,Nx,Ny,h)
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

   

    # Test reduction
    t_reduction = time()
    for _ in 1:iter_times
        # reduce(+,idata)
        sum(idata)
        # sum(idata_cpu)
    end
    synchronize()
    t_reduction = (time() - t_reduction ) * 1000 / iter_times
    @show t_reduction
    # End reduction test


    # Evaluating time in Data IO
    t_copy_data = time()
    iter_times_copy_data = 100
    for _ in 1:iter_times_copy_data
        copyto!(idata_cpu,idata)
    end
    # End evaluating time in Data IO
    t_copy_data = ( time() - t_copy_data ) * 1000 / iter_times_copy_data
    @show t_copy_data 

    
    t_boundary_data_part = time()
    iter_times_copy_data = 100
    for _ in 1:iter_times_copy_data
        # matrix_free_cpu_v4(idata,odata_cpu,Nx,Ny,h)
        matrix_free_cpu_optimized(idata,odata_boundary_GPU,Nx,Ny,h)
    end
    # End evaluating time in Data IO
    t_boundary_data_part = ( time() - t_boundary_data_part ) * 1000 / iter_times_copy_data
    @show t_boundary_data_part 


     # Evaluating final matrix-free A

   
    #  t_start_A = time()
    #  for _ in 1:iter_times
    #      matrix_free_A_v3(idata,odata,odata_D2_GPU,odata_boundary_GPU)
    #  end
    #  synchronize()
    #  t_A = (time() - t_start_A) * 1000 / iter_times
    #  @show t_A 
     # End evaluating matrix-free A in asynchrnous way

    #  iter_times = 1
    t_start_A_v4 = time()
    for _ in 1:iter_times
         matrix_free_A_v4(idata,odata)
    end
    synchronize()
    t_A_v4 = (time() - t_start_A_v4) * 1000 / iter_times
    @show t_A_v4 


    CUDA.unsafe_free!(idata)
    CUDA.unsafe_free!(odata)
    CUDA.unsafe_free!(odata_boundary_GPU)
    nothing
end





# test_matrix_free_A(13)
# test_matrix_free_A(14)
# test_matrix_free_A(15)