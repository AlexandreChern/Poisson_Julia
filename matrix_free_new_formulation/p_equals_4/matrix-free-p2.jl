function D2_split_dev_p2(idata,odata,coeff,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    # coeff = CuArray([1.0, 2.0, 1.0])
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    if 0 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = 0
    end

    if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
        odata[i,j] = (coeff[1] * idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
    end 

    nothing
end

function D2_matrix_free_p2_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1-1,TILE_DIM_1), div(Ny+TILE_DIM_2-1,TILE_DIM_2))
	blockdim = (TILE_DIM_1,TILE_DIM_2)

    coeff = cudaconvert(CuArray(Array([1.0,2.0,1.0])))

    @cuda threads=blockdim blocks=griddim D2_split_dev_p2(idata,odata,coeff,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    nothing
end