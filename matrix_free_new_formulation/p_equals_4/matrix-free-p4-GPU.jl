using CUDA

function matrix_free_D2_p4_GPU_kernel(idata,odata,coeff_d,Nx,Ny,hx,hy,::Val{TILE_DIM1},::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    if 0 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = 0
    end

    if 5 <= i <= Nx-4 && 5 <= j <= Ny - 4
        odata[i,j] = (coeff_d[1] * idata[i,j-2] + coeff_d[2] * idata[i,j-1] + coeff_d[3] * idata[i,j] + coeff_d[4] * idata[i,j+1] + coeff_d[5] * idata[i,j+2]
        + coeff_d[1] * idata[i-2,j] + coeff_d[2] * idata[i-1,j] + coeff_d[3] * idata[i,j] + coeff_d[4] * idata[i+1,j] + coeff_d[5] * idata[i+2,j]
    ) / hx^2
    end
    nothing
end

function matrix_free_D2_p4_GPU_kernel_v2(idata,odata,coeff_d,Nx,Ny,hx,hy,::Val{TILE_DIM1},::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    if 0 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = 0
    end

    if 5 <= i <= Nx-4 && 5 <= j <= Ny - 4
        odata[i,j] = -(-1/12 * idata[i,j-2] + 4/3 * idata[i,j-1] + -5/2 * idata[i,j] + 4/3 * idata[i,j+1] + -1/12 * idata[i,j+2]
        + -1/12 * idata[i-2,j] + 4/3 * idata[i-1,j] + -5/2 * idata[i,j] + 4/3 * idata[i+1,j] + -1/12 * idata[i+2,j]
    ) # removing calculation of division by hx^2
    # output -H_tilde * D2
    end
    nothing
end

function matrix_free_D2_p4_GPU(idata,odata)
    (Nx,Ny) = size(idata)
    hx=hy = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    coeff_d = CuArray([-1/12 4/3 -5/2 4/3 -1/12])
    @cuda threads=blockdim blocks=griddim matrix_free_D2_p4_GPU_kernel_v2(idata,odata,coeff_d,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)) # kernel_v2 for - H_tilde * D2
    nothing
end