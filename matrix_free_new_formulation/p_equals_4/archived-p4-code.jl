using CUDA

struct coef_GPU
    bhinv::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
    d::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
    bd::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
    BS::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}
end

function matrix_free_N_D2_kernel_1D_kernel(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM1}) where {TILE_DIM1}
    tidx = threadIdx().x
    # tidy = threadIdx().y

    j = (blockIdx().x - 1) * TILE_DIM1 + tidx
    # j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    # if 1 <= i <=4 && 1 <= j <= 4
    if 1 <= j <= 4
        for i in 1:4
            @inbounds odata[i,j] = - (coef_D.bd[j,1] * idata[i,1] + coef_D.bd[j,2] * idata[i,2] + coef_D.bd[j,3]*idata[i,3] + coef_D.bd[j,4]*idata[i,4] + coef_D.bd[j,5]*idata[i,5] + coef_D.bd[j,6] * idata[i,6]
        + coef_D.bd[i,1] * idata[1,j] + coef_D.bd[i,2] * idata[2,j] + coef_D.bd[i,3] * idata[3,j] + coef_D.bd[i,4]*idata[4,j] + coef_D.bd[i,5]*idata[5,j] + coef_D.bd[i,6] * idata[6,j]) / (coef_D.bhinv[i]*coef_D.bhinv[j]) # calculation for the left upper corner
        end

        for i in 1:4
            @inbounds odata[i,end+1-j] = - (coef_D.bd[j,1] * idata[i,end] + coef_D.bd[j,2] * idata[i,end-1] + coef_D.bd[j,3]*idata[i,end-2] + coef_D.bd[j,4]*idata[i,end-3] + coef_D.bd[j,5]*idata[i,end-4] + coef_D.bd[j,6] * idata[i,end-5]
        + coef_D.bd[i,1] * idata[1,end+1-j] + coef_D.bd[i,2] * idata[2,end+1-j] + coef_D.bd[i,3] * idata[3,end+1-j] + coef_D.bd[i,4]*idata[4,end+1-j] + coef_D.bd[i,5]*idata[5,end+1-j] + coef_D.bd[i,6] * idata[6,end+1-j]) / (coef_D.bhinv[i]*coef_D.bhinv[j]) # calculation for the left upper corner
        end
    end

    if 5 <= j <= Ny-4
        for i in 1:4
            @inbounds odata[i,j] = - (coef_D.d[1] * idata[i,j-2] + coef_D.d[2] * idata[i,j-1] + coef_D.d[3]*idata[i,j] + coef_D.d[4]*idata[i,j+1] + coef_D.d[5]*idata[i,j+2]
            + coef_D.bd[i,1] * idata[1,j] + coef_D.bd[i,2] * idata[2,j] + coef_D.bd[i,3] * idata[3,j] + coef_D.bd[i,4]*idata[4,j] + coef_D.bd[i,5]*idata[5,j] + coef_D.bd[i,6] * idata[6,j]) / (coef_D.bhinv[i]) # calculation for the upper edge
        end
    end
    nothing
end

function matrix_free_N_D2_kernel(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM1},::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    if 1 <= i <=4 && 1 <= j <= 4
            @inbounds odata[i,j] = - (coef_D.bd[j,1] * idata[i,1] + coef_D.bd[j,2] * idata[i,2] + coef_D.bd[j,3]*idata[i,3] + coef_D.bd[j,4]*idata[i,4] + coef_D.bd[j,5]*idata[i,5] + coef_D.bd[j,6] * idata[i,6]
        + coef_D.bd[i,1] * idata[1,j] + coef_D.bd[i,2] * idata[2,j] + coef_D.bd[i,3] * idata[3,j] + coef_D.bd[i,4]*idata[4,j] + coef_D.bd[i,5]*idata[5,j] + coef_D.bd[i,6] * idata[6,j]) / (coef_D.bhinv[i]*coef_D.bhinv[j]) # calculation for the left upper corner
    end

    if 1 <= i <= 4 && 1 <= j <= 4
        @inbounds odata[i,end+1-j] = - (coef_D.bd[j,1] * idata[i,end] + coef_D.bd[j,2] * idata[i,end-1] + coef_D.bd[j,3]*idata[i,end-2] + coef_D.bd[j,4]*idata[i,end-3] + coef_D.bd[j,5]*idata[i,end-4] + coef_D.bd[j,6] * idata[i,end-5]
        + coef_D.bd[i,1] * idata[1,end+1-j] + coef_D.bd[i,2] * idata[2,end+1-j] + coef_D.bd[i,3] * idata[3,end+1-j] + coef_D.bd[i,4]*idata[4,end+1-j] + coef_D.bd[i,5]*idata[5,end+1-j] + coef_D.bd[i,6] * idata[6,end+1-j]) / (coef_D.bhinv[i]*coef_D.bhinv[j]) # calculation for the left upper corner
    end

    if 1 <= i <= 4 && 5 <= j <= Ny-4
        @inbounds odata[i,j] = - (coef_D.d[1] * idata[i,j-2] + coef_D.d[2] * idata[i,j-1] + coef_D.d[3]*idata[i,j] + coef_D.d[4]*idata[i,j+1] + coef_D.d[5]*idata[i,j+2]
        + coef_D.bd[i,1] * idata[1,j] + coef_D.bd[i,2] * idata[2,j] + coef_D.bd[i,3] * idata[3,j] + coef_D.bd[i,4]*idata[4,j] + coef_D.bd[i,5]*idata[5,j] + coef_D.bd[i,6] * idata[6,j]) / (coef_D.bhinv[i]) # calculation for the upper edge
    end
    nothing
end

function matrix_free_N_D2_GPU_1D_kernel(idata,odata,coef_D,Nx,Ny,hx,hy) 
    TILE_DIM_1 = 256
    # griddim = (div(Nx+TILE_DIM_1+1,TILE_DIM_1),div(Ny+TILE_DIM_2-1,TILE_DIM_2))
    # blockdim = (TILE_DIM_1,TILE_DIM_2)

    griddim = div(Nx+TILE_DIM_1-1,TILE_DIM_1)
    blockdim = TILE_DIM_1
    @cuda threads=blockdim blocks=griddim matrix_free_N_D2_kernel_1D_kernel(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM_1))
end


function matrix_free_N_D2_GPU(idata,odata,coef_D,Nx,Ny,hx,hy) 
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx+TILE_DIM_1+1,TILE_DIM_1),div(Ny+TILE_DIM_2-1,TILE_DIM_2))
    blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim matrix_free_N_D2_kernel(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2))
end

Nx = Ny = 2^10+1

idata_GPU = CuArray(randn(Nx,Ny))
odata_GPU_N_D2 = CuArray(zeros(Nx,Ny))
odata_GPU_N_D2_v2 = CuArray(zeros(Nx,Ny))


# t_GPU_N_D2 = @elapsed for _ in 1:20000
#     matrix_free_N_D2_GPU(idata_GPU_N,odata_GPU_N_D2,coef_D,Nx,Ny,hx,hy)
# end

# t_GPU_N_D2_1D = @elapsed for _ in 1:20000
#     matrix_free_N_D2_GPU_1D_kernel(idata_GPU_N,odata_GPU_N_D2_v2,coef_D,Nx,Ny,hx,hy)
# end