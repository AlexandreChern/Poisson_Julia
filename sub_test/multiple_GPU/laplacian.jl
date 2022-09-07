using CUDA

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


function boundary(idata,odata)

end

