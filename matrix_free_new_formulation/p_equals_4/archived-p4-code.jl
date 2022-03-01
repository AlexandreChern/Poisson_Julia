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



function matrix_free_N_pseudo(idata,odata,Nx,Ny,hx,hy)
    tau_N = tau_S = -1;

    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];
    
    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];


    for j in 1:4
        if j == 1
            for i in 1:4
                odata[i,j] = ( - (bd[j,1] * idata[i,1] + bd[j,2] * idata[i,2] + bd[j,3]*idata[i,3] + bd[j,4]*idata[i,4] + bd[j,5]*idata[i,5] + bd[j,6] * idata[i,6]
                + bd[i,1] * idata[1,j] + bd[i,2] * idata[2,j] + bd[i,3] * idata[3,j] + bd[i,4]*idata[4,j] + bd[i,5]*idata[5,j] + bd[i,6] * idata[6,j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
                + -(-13*idata[i,j]/bhinv[i])  # only -H_tilde * tau_W*HI_x*E_W
                + -(beta*BS[j]/bhinv[i] * idata[i,1]) # only -H_tilde * beta*HI_x*BS_x'*E_W
                # + -(tau_N*BS[i]/bhinv[j] * idata[i,j]) # tau_N*HI_y*E_N*BS_y
                )

                odata[i,end+1-j] = ( - (bd[j,1] * idata[i,end] + bd[j,2] * idata[i,end-1] + bd[j,3]*idata[i,end-2] + bd[j,4]*idata[i,end-3] + bd[j,5]*idata[i,end-4] + bd[j,6] * idata[i,end-5]
                + bd[i,1] * idata[1,end+1-j] + bd[i,2] * idata[2,end+1-j] + bd[i,3] * idata[3,end+1-j] + bd[i,4]*idata[4,end+1-j] + bd[i,5]*idata[5,end+1-j] + bd[i,6] * idata[6,end+1-j]) / (bhinv[i]*bhinv[j]) # calculation for the left upper corner
                + -(-13*idata[i,end])/bhinv[i] 
                + -(beta*BS[j]/bhinv[i] * idata[i,end])
                # + -(tau_N*BS[i]/bhinv[j] * idata[i,end+1-j])
                )
            end
        end
    end
end
