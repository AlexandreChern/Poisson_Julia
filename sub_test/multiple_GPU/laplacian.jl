using CUDA

##################################################################################
#   Orientation in 2D:     #       Types of sub-domains
#   West: 1, East: 3       #       Leftmost: 1  Interior: 2 Rightmost: 3
#   South: 2, North: 4     #       
#                          #
#   ###### 4 ######        #       #########################################
#   #             #        #       #     |     |     |               |     #
#   #             #        #       #     |     |     |               |     #
#   1             3        #       #  1  |  2  | ... |               |  3  #
#   #             #        #       #     |     |     |               |     #
#   #             #        #       #     |     |     |               |     #
#   ###### 2 ######        #       #########################################
#                          #
##################################################################################

function copy_kernel(idata,odata,Nx,Ny)
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * blockDim().x + tidx
    j = (blockIdx().y - 1) * blockDim().y + tidy

    if 1 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = idata[i,j]
    end
    return nothing
end


function copy_GPU(idata,odata)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d copy_kernel(idata,odata,Nx,Ny)
end


function laplacian_GPU(idata,odata)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
    blockdim_2d = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim_2d blocks=griddim_2d laplacian_kernel(idata,odata,Nx,Ny)
end

function laplacian_kernel(idata,odata,Nx,Ny)
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * blockDim().x + tidx
    j = (blockIdx().y - 1) * blockDim().y + tidy

    if 1 <= i <= Nx && 1 <= j <= Ny
        if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
            odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
        else
            # odata[i,j] += 1
            odata[i,j] = 0
        end
    end
    return nothing
end


function boundary(idata,odata;orientation=1,type=1)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1D = 16
    blockdim_1D = TILE_DIM_1D
    griddim_1D = div(Nx,TILE_DIM_1D) + 1
    boundary_kernel = boundary_kernels[orientation,type]
    @cuda threads=blockdim_1D blocks=griddim_1D boundary_kernel_1_1(idata,odata,Nx,Ny)
end




function boundary_kernel_1_1(idata,odata,Nx,Ny)
    tidx = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tidx

    if 1 <= i <= Nx
        # odata[i,1] .= 1
        # odata[i,2] .= 2
        odata[i,1] = 1
        # odata[i,2] = 2
    end
    return nothing
end

boundary_kernels = [boundary_kernel_1_1,]


function


# boundary_1 = CuArray(randn(10,3))
# boundary_1_out = CuArray(zeros(10,3))

# boundary(boundary_1,view(boundary_1_out,:,1);orientation=1)
