using SparseArrays
using CUDA
using Random
function D2_test(idata,odata,Nx,Ny,h)
    for i = 1:Nx
        for j = 1:Ny
            global_index_x = (i-1)*Ny+j

            # idata_index_x = (i-1)*Ny + j + test_mod(i,Nx) * Nx
            # idata_index_y = (i-1)*Ny + j + test_mod(j,Ny)

            idata_index_x = (i-1)*Ny + j + (div(2*Nx-i-3,Nx-2) - 1) * Nx
            idata_index_y = (i-1)*Ny + j + div(2*Ny-j-3,Ny-2) - 1



            odata[global_index_x] = ( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny]) 
                                    + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) ) / h^2
        end
    end
end


function D2_test_v2(idata,odata,Nx,Ny,h)
    for i = 1:Nx
        for j = 1:Ny
            global_index_x = (i-1)*Ny+j

            # idata_index_x = (i-1)*Ny + j + test_mod(i,Nx) * Nx
            # idata_index_y = (i-1)*Ny + j + test_mod(j,Ny)

            offset_x = div(2*Nx-i-3,Nx-2) - 1
            offset_y = div(2*Ny-j-3,Ny-2) - 1

            idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
            idata_index_y = (i-1)*Ny + j + offset_y



            odata[global_index_x] = ( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny]) 
                                    + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) ) / 2^(abs(offset_x) + abs(offset_y))
        end
    end
end

function matrix_free_A_v2(idata,odata,Nx,Ny,h)
    # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    odata .= 0
    for i = 1:Nx
        for j = 1:Ny
            global_index = (i-1)*Ny+j

            # idata_index_x = (i-1)*Ny + j + test_mod(i,Nx) * Nx
            # idata_index_y = (i-1)*Ny + j + test_mod(j,Ny)

            offset_x = div(2*Nx-i-3,Nx-2) - 1
            offset_y = div(2*Ny-j-3,Ny-2) - 1

            idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
            idata_index_y = (i-1)*Ny + j + offset_y

            # odata[global_index] = 0


            odata[global_index] += ( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) ) / (h^2)  # D2


            odata[global_index] += abs(offset_y) * 2 * alpha1 * ( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y]) / h^2    # BS_y

            odata[global_index] += abs(offset_x) * ( beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h) # BS_x'
            odata[global_index+Ny*offset_x] += abs(offset_x) * beta * 2 * (-1*idata[global_index]) / h^2 # BS_x'
            odata[global_index+2*Ny*offset_x] += abs(offset_x) * beta * (0.5*idata[global_index]) / h^2 # BS_x'
        end
    end

end

function matrix_free_A_v4(idata,odata,Nx,Ny,h)
    # one function that does D2 + boundary conditions + H_tilde
    # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    odata .= 0
    offset_j = spzeros(Int,Ny)
    offset_j[2] = -1
    offset_j[3] = -2
    offset_j[Ny-1] = 1
    offset_j[Ny-2] = 2
    coef_j = spzeros(Ny)
    coef_j[1] = 2* 1.5
    coef_j[2] = - 2.0
    coef_j[3] = 0.5
    coef_j[Ny] = 2*1.5
    coef_j[Ny-1] = -2.0
    coef_j[Ny-2] = 0.5
    for i = 1:Nx
        for j = 1:Ny
            global_index = (i-1)*Ny+j

            # idata_index_x = (i-1)*Ny + j + test_mod(i,Nx) * Nx
            # idata_index_y = (i-1)*Ny + j + test_mod(j,Ny)

            offset_x = div(2*Nx-i-3,Nx-2) - 1
            offset_y = div(2*Ny-j-3,Ny-2) - 1

            idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
            idata_index_y = (i-1)*Ny + j + offset_y

            # odata[global_index] = 0


            # odata[global_index] = (( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) ) / (h^2)  
            # + abs(offset_y) * 2 * (alpha1 * (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y]) / h^2  
            # + abs(offset_x) * alpha4 * 2 * (idata[global_index] / h)  + coef_j[i] * beta * (idata[global_index+offset_j[i] * Ny]) / h^2)

            # Incorporating H_tilde
            odata[global_index] = (( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) )   
            + abs(offset_y) * 2 * alpha1 *( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y])
            + abs(offset_x) * alpha4 * 2 * (idata[global_index] * h)  + coef_j[i] * beta * (idata[global_index+offset_j[i] * Ny]) ) / 2^(abs(offset_x) + abs(offset_y))

            #####
            # form boundary conditions separately

            # odata[global_index] += abs(offset_y) * 2 * (alpha1 * (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y]) / h^2 # BS_y
            # odata[global_index] += abs(offset_x) * alpha4 * 2 * (idata[global_index] / h) # 
            # odata[global_index] += coef_j[i] * beta * (idata[global_index+offset_j[i] * Ny]) / h^2

            # # odata[global_index] += abs(offset_x) * ( beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h) # BS_x'
            # # odata[global_index+Ny*offset_x] += abs(offset_x) * beta * 2 * (-1*idata[global_index]) / h^2
            # # odata[global_index+2*Ny*offset_x] += abs(offset_x) * beta * (0.5*idata[global_index]) / h^2
        end
    end

end

function matrix_free_A_v5(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    # one function that does D2 + boundary conditions + H_tilde
    # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    odata .= 0
    offset_j = spzeros(Int,Ny)
    offset_j[2] = -1
    offset_j[3] = -2
    offset_j[Ny-1] = 1
    offset_j[Ny-2] = 2
    coef_j = spzeros(Ny)
    coef_j[1] = 2* 1.5
    coef_j[2] = - 2.0
    coef_j[3] = 0.5
    coef_j[Ny] = 2*1.5
    coef_j[Ny-1] = -2.0
    coef_j[Ny-2] = 0.5
    for i = 1:Nx
        for j = 1:Ny
            global_index = (i-1)*Ny+j

            offset_x = div(2*Nx-i-3,Nx-2) - 1
            offset_y = div(2*Ny-j-3,Ny-2) - 1

            idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
            idata_index_y = (i-1)*Ny + j + offset_y

            odata[global_index] = (( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) )   
            + abs(offset_y) * 2 * alpha1 * ( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y])
            + abs(offset_x) * alpha4 * 2 * (idata[global_index] * h) 
             + coef_j[i] * beta * (idata[global_index+offset_j[i] * Ny])
             ) / 2^(abs(offset_x) + abs(offset_y))

        end
    end
end


function matrix_free_GPU(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta,offset_j,coef_j,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    offset_x = div(2*Nx-i-3,Nx-2) - 1
    offset_y = div(2*Ny-j-3,Ny-2) - 1
    idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
    idata_index_y = (i-1)*Ny + j + offset_y

    # tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
    # tidy = (blockIdx().y - 1) * TILE_DIM + threadIdx().y

    # global_index = (tidx - 1) * Ny + tidy

    alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    # odata .= 0
    # offset_j = spzeros(Int,Ny)
    # offset_j = zeros(Int,Ny)
    # offset_j[2] = -1
    # offset_j[3] = -2
    # offset_j[Ny-1] = 1
    # offset_j[Ny-2] = 2
    # # coef_j = spzeros(Ny)
    # coef_j = zeros(Ny)
    # coef_j[1] = 2* 1.5
    # coef_j[2] = - 2.0
    # coef_j[3] = 0.5
    # coef_j[Ny] = 2*1.5
    # coef_j[Ny-1] = -2.0
    # coef_j[Ny-2] = 0.5

    if 1 <= i <= Nx && 1 <= j <= Ny
       

        odata[global_index] = (( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) )   
        + abs(offset_y) * 2 * alpha1 * ( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y])
        + abs(offset_x) * alpha4 * 2 * (idata[global_index] * h)  + coef_j[i] * beta * (idata[global_index+offset_j[i] * Ny]) ) / 2^(abs(offset_x) + abs(offset_y))      
    end
    nothing
end


function matrix_free_GPU_debug(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta,offset_j,coef_j,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    offset_x = div(2*Nx-i-3,Nx-2) - 1
    offset_y = div(2*Ny-j-3,Ny-2) - 1
    idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
    idata_index_y = (i-1)*Ny + j + offset_y

    # tidx = (blockIdx().x - 1) * TILE_DIM + threadIdx().x
    # tidy = (blockIdx().y - 1) * TILE_DIM + threadIdx().y

    # global_index = (tidx - 1) * Ny + tidy

    alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    # odata .= 0
    # offset_j = spzeros(Int,Ny)
    # offset_j = zeros(Int,Ny)
    # offset_j[2] = -1
    # offset_j[3] = -2
    # offset_j[Ny-1] = 1
    # offset_j[Ny-2] = 2
    # # coef_j = spzeros(Ny)
    # coef_j = zeros(Ny)
    # coef_j[1] = 2* 1.5
    # coef_j[2] = - 2.0
    # coef_j[3] = 0.5
    # coef_j[Ny] = 2*1.5
    # coef_j[Ny-1] = -2.0
    # coef_j[Ny-2] = 0.5

    if 1 <= i <= Nx && 1 <= j <= Ny
    # if global_index <= Nx*Ny
       
        odata[global_index] = 0
        odata[global_index] = (( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) )
        + abs(offset_y) * 2 * alpha1 * ( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y])
        + abs(offset_x) * alpha4 * 2 * (idata[global_index] * h) 
         + coef_j[i] * beta * idata[global_index+offset_j[i]* Ny]  # * (idata[global_index+offset_j[i] * Ny])   # this line is causing issue
         ) / 2^(abs(offset_x) + abs(offset_y))   
    end
    nothing
end

function matrix_free_GPU_debug_v2(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta,offset_j,coef_j,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    # offset_x = div(2*Nx-i-3,Nx-2) - 1
    # offset_y = div(2*Ny-j-3,Ny-2) - 1
    # idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
    # idata_index_y = (i-1)*Ny + j + offset_y


    # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    # odata .= 0
    # offset_j = spzeros(Int,Ny)
    # offset_j = zeros(Int,Ny)
    # offset_j[2] = -1
    # offset_j[3] = -2
    # offset_j[Ny-1] = 1
    # offset_j[Ny-2] = 2
    # # coef_j = spzeros(Ny)
    # coef_j = zeros(Ny)
    # coef_j[1] = 2* 1.5
    # coef_j[2] = - 2.0
    # coef_j[3] = 0.5
    # coef_j[Ny] = 2*1.5
    # coef_j[Ny-1] = -2.0
    # coef_j[Ny-2] = 0.5

    if 2 <= i <= Nx-1 && 2 <= j <= Ny-1
    # if global_index <= Nx*Ny
       
        odata[global_index] = 0
        odata[global_index] = (( (idata[global_index-Ny] - 2*idata[global_index] + idata[global_index + Ny])  + (idata[global_index-1] - 2*idata[global_index] + idata[global_index + 1]) )
        # + abs(offset_y) * 2 * (alpha1 * (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y])
        # + abs(offset_x) * alpha4 * 2 * (idata[global_index] * h) 
        #  + coef_j[i] * beta * idata[global_index+offset_j[i]* Ny]  # * (idata[global_index+offset_j[i] * Ny])   # this line is causing issue
        ) #  ) / 2^(abs(offset_x) + abs(offset_y))   
    end
    nothing
end

function Boundary_Conditions(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    odata .= 0
    for i = 1:Nx
        for j = 1:Ny
            global_index = (i-1)*Ny + j
            if j == Ny # N
                odata[global_index] += alpha1 * 2 * (1.5*idata[global_index] - 2*idata[global_index-1] + 0.5*idata[global_index-2]) / h^2
                # odata[global_index] = alpha1 * 2 * (1.5*idata[global_index] - 2*idata[global_index-1] + 0.5*idata[global_index-2]) / h^2
            end

            if j == 1 # S
                odata[global_index] += alpha2 * 2 *(1.5*idata[global_index] - 2*idata[global_index+1] + 0.5*idata[global_index+2]) / h^2
            end

            if i == Nx # E
                odata[global_index] += beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h
                odata[global_index-Ny] += beta * 2 * (-1*idata[global_index]) / h^2
                odata[global_index-2*Ny] += beta * (0.5*idata[global_index]) / h^2
            end

            if i == 1 # W
                odata[global_index] += beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha3 * 2* (idata[global_index]) / h
                odata[global_index+Ny] += beta * 2 * (-1*idata[global_index]) / h^2
                odata[global_index+2*Ny] += beta * (0.5*idata[global_index]) / h^2
            end
        end
    end
end

function Boundary_Conditions_v2(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    odata .= 0
    for i = 1:Nx
        for j = 1:Ny
            global_index = (i-1)*Ny + j
            offset_x = div(2*Nx-i-3,Nx-2) - 1
            offset_y = div(2*Ny-j-3,Ny-2) - 1
            # if j == Ny # N
            #     odata[global_index] += alpha1 * 2 * (1.5*idata[global_index] - 2*idata[global_index-1] + 0.5*idata[global_index-2]) / (2^(abs(offset_x)+abs(offset_y)))
            #     # odata[global_index] = alpha1 * 2 * (1.5*idata[global_index] - 2*idata[global_index-1] + 0.5*idata[global_index-2]) / h^2
            # end

            # if j == 1 # S
            #     odata[global_index] += alpha2 * 2 *(1.5*idata[global_index] - 2*idata[global_index+1] + 0.5*idata[global_index+2]) / h^2
            # end
            odata[global_index] += abs(offset_y) * 2 * alpha1 * ( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y]) / h^2# + abs(offset_x) * ( beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h)
                                # +  abs(offset_x) * ( beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h)

            # if i == Nx # E
            #     odata[global_index] += beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h
            #     odata[global_index-Ny] += beta * 2 * (-1*idata[global_index]) / h^2
            #     odata[global_index-2*Ny] += beta * (0.5*idata[global_index]) / h^2
            # end

            # if i == 1 # W
            #     odata[global_index] += beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha3 * 2* (idata[global_index]) / h
            #     odata[global_index+Ny] += beta * 2 * (-1*idata[global_index]) / h^2
            #     odata[global_index+2*Ny] += beta * (0.5*idata[global_index]) / h^2
            # end
            odata[global_index] += abs(offset_x) * ( beta * 2 * (1.5*idata[global_index] ) / h^2 + alpha4 * 2* (idata[global_index]) / h)
            odata[global_index+Ny*offset_x] += abs(offset_x) * beta * 2 * (-1*idata[global_index]) / h^2
            odata[global_index+2*Ny*offset_x] += abs(offset_x) * beta * (0.5*idata[global_index]) / h^2
        end
    end
end

function H_tilde_matrix_free(idata,odata,Nx,Ny,h)
    for i in 1:Nx
        for j in 1:Ny
            global_index = (i-1)*Ny+j
            odata[global_index] = idata[global_index] * h^2
            if i == 1
                odata[global_index] /= 2
            end
            if j == 1
                odata[global_index] /= 2
            end
            if i == Nx
                odata[global_index] /= 2
            end
            if j == Ny
                odata[global_index] /= 2
            end
        end
    end
end

function H_tilde_matrix_free_v2(idata,odata,Nx,Ny,h)
    for i in 1:Nx
        for j in 1:Ny
            global_index = (i-1)*Ny + j
            var = abs(div(2*Nx-i-3,Nx-2) - 1) + abs(div(2*Ny-j-3,Ny-2)-1)
            odata[global_index] = idata[global_index] * h^2 / (2^var)
            # odata[global_index] = var
        end
    end
end



function test_H_tilde(Nx,Ny)
    h = 1/(Nx-1)
    # idata = randn(Nx*Ny)
    idata = ones(Nx*Ny)
    odata1 = spzeros(Nx*Ny)
    odata2 = spzeros(Nx*Ny)
    H_tilde_matrix_free(idata,odata1,Nx,Ny,h)
    H_tilde_matrix_free_v2(idata,odata2,Nx,Ny,h)
    # @show odata1
    # @show odata2
    @show odata1 - odata2
end

function test_D2(Nx,Ny)
    h = 1/(Nx-1)
    # idata = randn(Nx*Ny)
    idata = ones(Nx*Ny)
    odata1 = spzeros(Nx*Ny)
    odata2 = spzeros(Nx*Ny)
    odata3 = spzeros(Nx*Ny)
    D2_test(idata,odata1,Nx,Ny,h)
    H_tilde_matrix_free(odata1,odata2,Nx,Ny,h)
    D2_test_v2(idata,odata3,Nx,Ny,h)
    # @show odata1
    # @show odata2
    @show odata2 - odata3
end

function test_Boundary_Conditions(Nx,Ny)
    h = 1/(Nx-1)
    alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    # idata = randn(Nx*Ny)
    idata = randn(Nx*Ny)
    odata1 = spzeros(Nx*Ny)
    odata2 = spzeros(Nx*Ny)
    Boundary_Conditions(idata,odata1,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    Boundary_Conditions_v2(idata,odata2,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    # @show odata1
    # @show odata2
    @show odata1 - odata2
end

function matrix_free_A(idata,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    odata1 = spzeros(Nx*Ny)
    odata2 = spzeros(Nx*Ny)
    D2_test(idata,odata1,Nx,Ny,h)
    Boundary_Conditions(idata,odata2,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    # odata3 = odata1 .+ odata2
    # return odata
    odata .= odata1 .+ odata2
    H_tilde_matrix_free(odata,odata,Nx,Ny,h)
    # return odata3
end


function test_matrix_free(Nx,Ny)
    h = 1/(Nx-1)
    alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    alpha1 = alpha2 = -13/h
    idata = randn(Nx*Ny)
    odata1 = spzeros(Nx*Ny)
    odata2 = spzeros(Nx*Ny)
    odata3 = spzeros(Nx*Ny)
    odata4 = spzeros(Nx*Ny)
    odata5 = spzeros(Nx*Ny)
    odata6 = spzeros(Nx*Ny)
    D2_test(idata,odata1,Nx,Ny,h)
    Boundary_Conditions(idata,odata2,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    # D2_test_v2(idata,odata3,Nx,Ny,h)
    Boundary_Conditions_v2(idata,odata4,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    matrix_free_A(idata,odata3,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    matrix_free_A_v2(idata,odata5,Nx,Ny,h) # without incorporating H_tilde
    matrix_free_A_v4(idata,odata6,Nx,Ny,h) # including H_tilde
    @show reshape(odata1 + odata2 - odata5,Nx,Ny)
    @show reshape(odata3 - odata6,Nx,Ny)
    @assert odata1 + odata2 ≈ odata5
    @assert odata3 ≈ odata6
end


function test_GPU_kernel(level)
    alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
    Nx = Ny = 2^level + 1
    h = 1/(Nx-1)
    Random.seed!(0)
    u = randn(Nx*Ny)
    odata = randn(Nx*Ny)
    d_u = CuArray(u)
    d_odata = CuArray(odata)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

    offset_j = CuArray(zeros(Int,Ny))
    # offset_j = zeros(Int,Ny)
    offset_j[2] = -1
    offset_j[3] = -2
    offset_j[Ny-1] = 1
    offset_j[Ny-2] = 2
    # coef_j = spzeros(Ny)
    coef_j = CuArray(zeros(Float64,Ny))
    coef_j[1] = 2* 1.5
    coef_j[2] = - 2.0
    coef_j[3] = 0.5
    coef_j[Ny] = 2*1.5
    coef_j[Ny-1] = -2.0
    coef_j[Ny-2] = 0.5
    matrix_free_A_v5(u,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    @cuda threads=blockdim blocks=griddim matrix_free_GPU_debug(d_u,d_odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta,offset_j,coef_j,Val(TILE_DIM_1), Val(TILE_DIM_2))
    # @show reshape(odata,Nx,Ny)
    # @show reshape(d_odata,Nx,Ny)

    @assert odata ≈ Array(d_odata)

    iter_times = 100

    t0 = time()
    for i in 1:iter_times
        matrix_free_A_v5(u,odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta)
    end
    t_cpu = time() - t0

    t1 = time()
    for i in 1:iter_times
        @cuda threads=blockdim blocks=griddim matrix_free_GPU_debug(d_u,d_odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta,offset_j,coef_j,Val(TILE_DIM_1), Val(TILE_DIM_2))
    end
    synchronize()
    t_gpu = time() - t1

    t_D2 = time()
    for i in 1:iter_times
        @cuda threads=blockdim blocks=griddim matrix_free_GPU_debug_v2(d_u,d_odata,Nx,Ny,h,alpha1,alpha2,alpha3,alpha4,beta,offset_j,coef_j,Val(TILE_DIM_1), Val(TILE_DIM_2))
    end
    synchronize()
    t_D2 = time() - t_D2

    @show t_cpu
    @show t_gpu
    @show t_D2
end

function test_sparse_matrix(A,b)
    iter_times = 1000
    t3 = time()
    for i in 1:iter_times
        out = A*b
    end
    cpu_time = time() - t3
    @show cpu_time 
end