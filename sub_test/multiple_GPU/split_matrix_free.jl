using CUDA


function matrix_free_A(idata,odata,Nx,Ny,type,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    if type == 1
        if 1 <= i <= Nx && 1 <= j <= Ny
            odata[i,j] = 0
        end

        if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
            @inbounds   odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
        end 
    else
        if 1 <= i <= Nx && 1 <= j <= Ny
            odata[i,j] = 0
        end
    end
    nothing
end