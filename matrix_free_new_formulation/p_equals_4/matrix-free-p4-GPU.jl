using CUDA
using Adapt

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

    # if 0 <= i <= Nx && 1 <= j <= Ny
    #     odata[i,j] = 0
    # end

    if 5 <= i <= Nx-4 && 5 <= j <= Ny - 4
        @inbounds odata[i,j] = -(-1/12 * idata[i,j-2] + 4/3 * idata[i,j-1] + -5/2 * idata[i,j] + 4/3 * idata[i,j+1] + -1/12 * idata[i,j+2]
        + -1/12 * idata[i-2,j] + 4/3 * idata[i-1,j] + -5/2 * idata[i,j] + 4/3 * idata[i+1,j] + -1/12 * idata[i+2,j]
    ) # removing calculation of division by hx^2
    # output -H_tilde * D2
    end
    nothing
end

function matrix_free_D2_p4_GPU_kernel_v3(idata,odata,Nx,Ny,hx,hy,::Val{TILE_DIM1},::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    # if 0 <= i <= Nx && 1 <= j <= Ny
    #     odata[i,j] = 0
    # end

    if 5 <= i <= Nx-4 && 5 <= j <= Ny - 4
        @inbounds odata[i,j] = -(-1/12 * idata[i,j-2] + 4/3 * idata[i,j-1] + -5/2 * idata[i,j] + 4/3 * idata[i,j+1] + -1/12 * idata[i,j+2]
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
    # coeff_d = cudaconvert(CuArray([-1/12 4/3 -5/2 4/3 -1/12]))
    # @cuda threads=blockdim blocks=griddim matrix_free_D2_p4_GPU_kernel_v2(idata,odata,coeff_d,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)) # kernel_v2 for - H_tilde * D2
    @cuda threads=blockdim blocks=griddim matrix_free_D2_p4_GPU_kernel_v3(idata,odata,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)) # kernel_v2 for - H_tilde * D2
    nothing
end





struct coef_GPU_v3{A}
    bhinv::A
    d::A
    bd::A
    BS::A
end

Adapt.@adapt_structure coef_GPU_v3

coef = coef_GPU_v3(CuArray([48/17 48/59 48/43 48/49]),
                CuArray([-1/12 4/3 -5/2 4/3 -1/12]),
                CuArray([ 2    -5       4     -1       0      0;
                    1    -2       1      0       0      0;
                    -4/43 59/43 -110/43  59/43   -4/43   0;
                    -1/49  0      59/49 -118/49  64/49  -4/49]),
                CuArray([11/6 -3 3/2 -1/3]))

isbits(coef)
isbits(cudaconvert(coef))

coef_D = cudaconvert(coef)

# coef = coef_GPU_v3(CuArray([48/17 48/59 48/43 48/49]),
#                 CuArray([-1/12 4/3 -5/2 4/3 -1/12]),
#                 CuArray([ 2    -5       4     -1       0      0;
#                     1    -2       1      0       0      0;
#                     -4/43 59/43 -110/43  59/43   -4/43   0;
#                     -1/49  0      59/49 -118/49  64/49  -4/49]),
#                 CuArray([11/6 -3 3/2 -1/3]))


isbits(cudaconvert(coef))

function _matrix_free_N_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    # tidy = threadIdx().y

    j = (blockIdx().x - 1) * TILE_DIM + tidx
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

function matrix_free_N_D2_GPU(idata,odata,coef_D,Nx,Ny,hx,hy) 
    TILE_DIM = 256

    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_N_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end

function _matrix_free_S_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    j = (blockIdx().x - 1) * TILE_DIM + tidx

    if 1 <= j <= 4
        for i in 1:4
            @inbounds odata[end+1-i,j] = - (coef_D.bd[j,1] * idata[end+1-i,1] + coef_D.bd[j,2] * idata[end+1-i,2] + coef_D.bd[j,3]*idata[end+1-i,3] + coef_D.bd[j,4]*idata[end+1-i,4] + coef_D.bd[j,5]*idata[end+1-i,5] + coef_D.bd[j,6] * idata[end+1-i,6]
            + coef_D.bd[i,1] * idata[end,j] + coef_D.bd[i,2] * idata[end-1,j] + coef_D.bd[i,3] * idata[end-2,j] + coef_D.bd[i,4]*idata[end-3,j] + coef_D.bd[i,5]*idata[end-4,j] + coef_D.bd[i,6] * idata[end-5,j]) / (coef_D.bhinv[i]*coef_D.bhinv[j]) # calculation for the left lower corner

            @inbounds odata[end+1-i,end+1-j] = -(coef_D.bd[j,1]*idata[end+1-i,end] + coef_D.bd[j,2]*idata[end+1-i,end-1] + coef_D.bd[j,3]*idata[end+1-i,end-2] + coef_D.bd[j,4]*idata[end+1-i,end-3] + coef_D.bd[j,5]*idata[end+1-i,end-4] + coef_D.bd[j,6]*idata[end+1-i,end-5]
            + coef_D.bd[i,1] * idata[end,end+1-j] + coef_D.bd[i,2] * idata[end-1,end+1-j] + coef_D.bd[i,3] * idata[end-2,end+1-j] + coef_D.bd[i,4]*idata[end-3,end+1-j] + coef_D.bd[i,5]*idata[end-4,end+1-j] + coef_D.bd[i,6] * idata[end-5,end+1-j]) / (coef_D.bhinv[i]*coef_D.bhinv[j]) # calculation for the right lower corner
        end
    end

    if 5 <= j <= Nx-4
        for i in 1:4
            @inbounds odata[end+1-i,j] = - (coef_D.d[1] * idata[end+1-i,j-2] + coef_D.d[2] * idata[end+1-i,j-1] + coef_D.d[3]*idata[end+1-i,j] + coef_D.d[4]*idata[end+1-i,j+1] + coef_D.d[5]*idata[end+1-i,j+2]
            + coef_D.bd[i,1] * idata[end,j] + coef_D.bd[i,2] * idata[end-1,j] + coef_D.bd[i,3] * idata[end-2,j] + coef_D.bd[i,4]*idata[end-3,j] + coef_D.bd[i,5]*idata[end-4,j] + coef_D.bd[i,6] * idata[end-5,j]) / (coef_D.bhinv[i]) # calculation for the lower edge
        end
    end

    nothing
end

function matrix_free_S_D2_GPU(idata,odata,coef_D,Nx,Ny,hx,hy)
    TILE_DIM = 256
    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_S_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end


function _matrix_free_E_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    i = (blockIdx().x - 1) * TILE_DIM + tidx

    if 1 <= i <= 4
        for j in 1:4
            # @inbounds odata[i,j] = 0
            @inbounds odata[i,end+1-j] = 0
            # @inbounds odata[end+1-i,1] = 0
            @inbounds odata[end+1-i,end+1-j] = 0
        end
    end

    if 5 <= i <= Ny-4
        for j in 1:4
            @inbounds odata[i,end+1-j] = - (coef_D.d[1]*idata[i-2,end+1-j] + coef_D.d[2] * idata[i-1,end+1-j] + coef_D.d[3] * idata[i,end+1-j] + coef_D.d[4] * idata[i+1,end+1-j] + coef_D.d[5]*idata[i+2,end+1-j]
            + coef_D.bd[j,1]*idata[i,end] + coef_D.bd[j,2]*idata[i,end-1] + coef_D.bd[j,3] * idata[i,end-2] + coef_D.bd[j,4]*idata[i,end-3] + coef_D.bd[j,5]*idata[i,end-4] + coef_D.bd[j,6]*idata[i,end-5]) / coef_D.bhinv[j]
        end
    end
end

function matrix_free_E_D2_GPU(idata,odata,coef_D,Nx,Ny,hx,hy)
    TILE_DIM = 256
    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_E_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end

function _matrix_free_W_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    i = (blockIdx().x - 1) * TILE_DIM + tidx

    if 1 <= i <= 4
        for j in 1:4
            @inbounds odata[i,j] = 0
            # @inbounds odata[i,end+1-j] = 0
            @inbounds odata[end+1-i,1] = 0
            # @inbounds odata[end+1-i,end+1-j] = 0
        end
    end

    if 5 <= i <= Ny-4
        for j in 1:4
            @inbounds odata[i,j] = - (coef_D.d[1]*idata[i-2,j] + coef_D.d[2] * idata[i-1,j] + coef_D.d[3] * idata[i,j] + coef_D.d[4] * idata[i+1,j] + coef_D.d[5]*idata[i+2,j]
            + coef_D.bd[j,1]*idata[i,1] + coef_D.bd[j,2]*idata[i,2] + coef_D.bd[j,3] * idata[i,3] + coef_D.bd[j,4]*idata[i,4] + coef_D.bd[j,5]*idata[i,5] + coef_D.bd[j,6]*idata[i,6]) / coef_D.bhinv[j]
        end
    end
end

function matrix_free_W_D2_GPU(idata,odata,coef_D,Nx,Ny,hx,hy)
    TILE_DIM = 256
    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_W_D2_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end


## Matrix free boundary Operators

function _matrix_free_N_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    j = (blockIdx().x - 1) * TILE_DIM + tidx
    tau_N = -1
    if 1 <= j <= 4
        odata[1,j] = 0
        odata[1,end+1-j] = 0
        for i in 1:4
            odata[1,j] += -(tau_N*coef_D.BS[i]/coef_D.bhinv[j] * idata[i,j]) # tau_N*HI_y*E_N*BS_y
            odata[1,end+1-j] += -(tau_N*coef_D.BS[i]/coef_D.bhinv[j] * idata[i,end+1-j])
        end
    end

    if 5 <= j <= Nx-4
        odata[1,j] = 0
        for i in 1:4
            odata[1,j] += -(tau_N*coef_D.BS[i]*idata[i,j])
        end
    end
    nothing
end

function matrix_free_N_P_GPU(idata,odata,coef_D,Nx,Ny,hx,hy)
    TILE_DIM = 256
    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_N_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end

function _matrix_free_S_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    j = (blockIdx().x - 1) * TILE_DIM + tidx
    tau_S = -1
    if 1 <= j <= 4
        odata[end,j] = 0
        odata[end,end+1-j] = 0
        for i in 1:4
            odata[end,j] += -(tau_S*coef_D.BS[i]/coef_D.bhinv[j] * idata[end+1-i,j]) # tau_N*HI_y*E_N*BS_y
            odata[end,end+1-j] += -(tau_S*coef_D.BS[i]/coef_D.bhinv[j] * idata[end+1-i,end+1-j])
        end
    end

    if 5 <= j <= Nx-4
        odata[end,j] = 0
        for i in 1:4
            odata[end,j] += -(tau_S*coef_D.BS[i]*idata[end+1-i,j])
        end
    end
    nothing
end

function matrix_free_S_P_GPU(idata,odata,coef_D,Nx,Ny,hx,hy)
    TILE_DIM = 256
    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_S_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end


function _matrix_free_W_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    i = (blockIdx().x - 1) * TILE_DIM + tidx
    beta = 1

    if 1 <= i <= Ny
        for j in 1:4
            odata[i,j] = 0 
        end
    end

    if 1 <= i <= 4
        odata[i,1] = -(-13*idata[i,1]/coef_D.bhinv[i])
        odata[end+1-i,1] = -(-13*idata[end+1-i,1]/coef_D.bhinv[i])
        for j = 1:4
            odata[i,j] += -(beta*coef_D.BS[j]/coef_D.bhinv[i] * idata[i,1])
            odata[end+1-i,j] += -(beta*coef_D.BS[j]/coef_D.bhinv[i] * idata[end+1-i,1])
        end
    end

    if 5 <= i <= Ny-4
        odata[i,1] = -(-13*idata[i,1])
        for j in 1:4
            odata[i,j] += -(beta*coef_D.BS[j] * idata[i,1])
        end
    end
    nothing
end

function matrix_free_W_P_GPU(idata,odata,coef_D,Nx,Ny,hx,hy)
    TILE_DIM = 256
    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_W_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end



function _matrix_free_E_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,::Val{TILE_DIM}) where {TILE_DIM}
    tidx = threadIdx().x
    i = (blockIdx().x - 1) * TILE_DIM + tidx
    beta = 1

    if 1 <= i <= Ny
        for j in 1:4
            odata[i,j] = 0 
        end
    end

    if 1 <= i <= 4
        odata[i,end] = -(-13*idata[i,end]/coef_D.bhinv[i])
        odata[end+1-i,end] = -(-13*idata[end+1-i,end]/coef_D.bhinv[i])
        for j = 1:4
            odata[i,end+1-j] += -(beta*coef_D.BS[j]/coef_D.bhinv[i] * idata[i,end])
            odata[end+1-i,end+1-j] += -(beta*coef_D.BS[j]/coef_D.bhinv[i] * idata[end+1-i,end])
        end
    end

    if 5 <= i <= Ny-4
        odata[i,end] = -(-13*idata[i,end])
        for j in 1:4
            odata[i,end+1-j] += -(beta*coef_D.BS[j] * idata[i,end])
        end
    end
    nothing
end

function matrix_free_E_P_GPU(idata,odata,coef_D,Nx,Ny,hx,hy)
    TILE_DIM = 256
    griddim = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim = TILE_DIM
    @cuda threads=blockdim blocks=griddim _matrix_free_E_P_kernel_(idata,odata,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
end




function matrix_free_HA_GPU(idata_GPU,odata_GPU,coef_D,Nx,Ny,hx,hy)

    idata_GPU_N = @view idata_GPU[1:6,1:end]
    idata_GPU_S = @view idata_GPU[end-5:end,1:end]
    idata_GPU_W = @view idata_GPU[1:end,1:6]
    idata_GPU_E = @view idata_GPU[1:end,end-5:end]


    odata_GPU_N_D2 = CuArray(zeros(4,Ny))
    odata_GPU_S_D2 = CuArray(zeros(4,Ny))
    odata_GPU_W_D2 = CuArray(zeros(Ny,4))
    odata_GPU_E_D2 = CuArray(zeros(Ny,4))
    
    odata_GPU_N_P = CuArray(zeros(4,Nx))
    odata_GPU_S_P = CuArray(zeros(4,Nx))
    odata_GPU_W_P = CuArray(zeros(Ny,4))
    odata_GPU_E_P = CuArray(zeros(Ny,4))

    # odata_GPU = CuArray(zeros(Ny,Nx))
    odata_GPU .= 0
    
    matrix_free_D2_p4_GPU(idata_GPU,odata_GPU)

    matrix_free_N_D2_GPU(idata_GPU_N,odata_GPU_N_D2,coef_D,Nx,Ny,hx,hy)

    matrix_free_S_D2_GPU(idata_GPU_S,odata_GPU_S_D2,coef_D,Nx,Ny,hx,hy)

    matrix_free_W_D2_GPU(idata_GPU_W,odata_GPU_W_D2,coef_D,Nx,Ny,hx,hy)

    matrix_free_E_D2_GPU(idata_GPU_E,odata_GPU_E_D2,coef_D,Nx,Ny,hx,hy)

    copyto!(view(odata_GPU,1:4,1:Nx),odata_GPU_N_D2)
    copyto!(view(odata_GPU,Ny-3:Ny,1:Nx),odata_GPU_S_D2)
    copyto!(view(odata_GPU,5:Ny-4,1:4),view(odata_GPU_W_D2,5:Ny-4,1:4))
    copyto!(view(odata_GPU,5:Ny-4,Nx-3:Nx),view(odata_GPU_E_D2,5:Ny-4,1:4))

    matrix_free_N_P_GPU(idata_GPU_N,odata_GPU_N_P,coef_D,Nx,Ny,hx,hy)

    matrix_free_S_P_GPU(idata_GPU_S,odata_GPU_S_P,coef_D,Nx,Ny,hx,hy)

    matrix_free_W_P_GPU(idata_GPU_W,odata_GPU_W_P,coef_D,Nx,Ny,hx,hy)

    matrix_free_E_P_GPU(idata_GPU_E,odata_GPU_E_P,coef_D,Nx,Ny,hx,hy)

    copyto!(view(odata_GPU,1:1,1:Nx),view(odata_GPU,1,1:Nx)+view(odata_GPU_N_P,1,1:Nx))

    copyto!(view(odata_GPU,Ny,1:Nx),view(odata_GPU,Ny,1:Nx)+view(odata_GPU_S_P,4,1:Nx))

    copyto!(view(odata_GPU,1:Ny,1:4),view(odata_GPU,1:Ny,1:4)+view(odata_GPU_W_P,1:Ny,1:4))
    copyto!(view(odata_GPU,1:Ny,Nx-3:Nx),view(odata_GPU,1:Ny,Nx-3:Nx)+view(odata_GPU_E_P,1:Ny,1:4))

end

struct odata_GPU_NSWE{T}
    odata_GPU_N_D2::T
    odata_GPU_S_D2::T
    odata_GPU_W_D2::T
    odata_GPU_E_D2::T

    odata_GPU_N_P::T
    odata_GPU_S_P::T
    odata_GPU_W_P::T
    odata_GPU_E_P::T
end



function matrix_free_HA_GPU_v2(idata_GPU,odata_GPU,odata_GPU_NSWE_tmp,coef_D,Nx,Ny,hx,hy)
    idata_GPU_N = @view idata_GPU[1:6,1:end]
    idata_GPU_S = @view idata_GPU[end-5:end,1:end]
    idata_GPU_W = @view idata_GPU[1:end,1:6]
    idata_GPU_E = @view idata_GPU[1:end,end-5:end]


    # odata_GPU_N_D2 = CuArray(zeros(4,Nx))
    # odata_GPU_S_D2 = CuArray(zeros(4,Nx))
    # odata_GPU_W_D2 = CuArray(zeros(Ny,4))
    # odata_GPU_E_D2 = CuArray(zeros(Ny,4))
    
    # odata_GPU_N_P = CuArray(zeros(4,Nx))
    # odata_GPU_S_P = CuArray(zeros(4,Nx))
    # odata_GPU_W_P = CuArray(zeros(Ny,4))
    # odata_GPU_E_P = CuArray(zeros(Ny,4))

    # Grid Information for GPU kernels
    # (Nx,Ny) = size(idata_GPU)
    hx=hy = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2D = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim_2D = (TILE_DIM_1,TILE_DIM_2)
    
    TILE_DIM = 256
    griddim_1D = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim_1D = TILE_DIM

    coeff_d = cudaconvert(CuArray([-1/12 4/3 -5/2 4/3 -1/12]))


    @cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p4_GPU_kernel_v2(idata_GPU,odata_GPU,coeff_d,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)) # kernel_v2 for - H_tilde * D2

    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_N_D2_kernel_(idata_GPU_N,odata_GPU_NSWE_tmp.odata_GPU_N_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_S_D2_kernel_(idata_GPU_S,odata_GPU_NSWE_tmp.odata_GPU_S_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_W_D2_kernel_(idata_GPU_W,odata_GPU_NSWE_tmp.odata_GPU_W_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_E_D2_kernel_(idata_GPU_E,odata_GPU_NSWE_tmp.odata_GPU_E_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))

    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_N_P_kernel_(idata_GPU_N,odata_GPU_NSWE_tmp.odata_GPU_N_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_S_P_kernel_(idata_GPU_S,odata_GPU_NSWE_tmp.odata_GPU_S_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_W_P_kernel_(idata_GPU_W,odata_GPU_NSWE_tmp.odata_GPU_W_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_E_P_kernel_(idata_GPU_E,odata_GPU_NSWE_tmp.odata_GPU_E_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM))

    copyto!(view(odata_GPU,1:4,1:Nx),odata_GPU_NSWE_tmp.odata_GPU_N_D2)
    copyto!(view(odata_GPU,Ny-3:Ny,1:Nx),odata_GPU_NSWE_tmp.odata_GPU_S_D2)
    copyto!(view(odata_GPU,5:Ny-4,1:4),view(odata_GPU_NSWE_tmp.odata_GPU_W_D2,5:Ny-4,1:4))
    copyto!(view(odata_GPU,5:Ny-4,Nx-3:Nx),view(odata_GPU_NSWE_tmp.odata_GPU_E_D2,5:Ny-4,1:4))

    copyto!(view(odata_GPU,1,1:Nx),view(odata_GPU,1,1:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_N_P,1,1:Nx))
    copyto!(view(odata_GPU,Ny,1:Nx),view(odata_GPU,Ny,1:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_S_P,4,1:Nx))
    copyto!(view(odata_GPU,1:Ny,1:4),view(odata_GPU,1:Ny,1:4).+view(odata_GPU_NSWE_tmp.odata_GPU_W_P,1:Ny,1:4))
    copyto!(view(odata_GPU,1:Ny,Nx-3:Nx),view(odata_GPU,1:Ny,Nx-3:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_E_P,1:Ny,1:4))

end



function matrix_free_HA_GPU_v3(idata_GPU,odata_GPU,odata_GPU_NSWE_tmp,coef_D,Nx,Ny,hx,hy)
    idata_GPU_N = @view idata_GPU[1:6,1:end]
    idata_GPU_S = @view idata_GPU[end-5:end,1:end]
    idata_GPU_W = @view idata_GPU[1:end,1:6]
    idata_GPU_E = @view idata_GPU[1:end,end-5:end]


    # odata_GPU_N_D2 = CuArray(zeros(4,Nx))
    # odata_GPU_S_D2 = CuArray(zeros(4,Nx))
    # odata_GPU_W_D2 = CuArray(zeros(Ny,4))
    # odata_GPU_E_D2 = CuArray(zeros(Ny,4))
    
    # odata_GPU_N_P = CuArray(zeros(4,Nx))
    # odata_GPU_S_P = CuArray(zeros(4,Nx))
    # odata_GPU_W_P = CuArray(zeros(Ny,4))
    # odata_GPU_E_P = CuArray(zeros(Ny,4))

    # Grid Information for GPU kernels
    # (Nx,Ny) = size(idata_GPU)
    hx=hy = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2D = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim_2D = (TILE_DIM_1,TILE_DIM_2)
    
    TILE_DIM = 256
    griddim_1D = div(Nx+TILE_DIM-1,TILE_DIM)
    blockdim_1D = TILE_DIM

    # coeff_d = cudaconvert(CuArray([-1/12 4/3 -5/2 4/3 -1/12]))


    @cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p4_GPU_kernel_v3(idata_GPU,odata_GPU,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)); # kernel_v2 for - H_tilde * D2

    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_N_D2_kernel_(idata_GPU_N,odata_GPU_NSWE_tmp.odata_GPU_N_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_S_D2_kernel_(idata_GPU_S,odata_GPU_NSWE_tmp.odata_GPU_S_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_W_D2_kernel_(idata_GPU_W,odata_GPU_NSWE_tmp.odata_GPU_W_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_E_D2_kernel_(idata_GPU_E,odata_GPU_NSWE_tmp.odata_GPU_E_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));

    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_N_P_kernel_(idata_GPU_N,odata_GPU_NSWE_tmp.odata_GPU_N_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_S_P_kernel_(idata_GPU_S,odata_GPU_NSWE_tmp.odata_GPU_S_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_W_P_kernel_(idata_GPU_W,odata_GPU_NSWE_tmp.odata_GPU_W_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_E_P_kernel_(idata_GPU_E,odata_GPU_NSWE_tmp.odata_GPU_E_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));

    copyto!(view(odata_GPU,1:4,1:Nx),odata_GPU_NSWE_tmp.odata_GPU_N_D2);
    copyto!(view(odata_GPU,Ny-3:Ny,1:Nx),odata_GPU_NSWE_tmp.odata_GPU_S_D2);
    copyto!(view(odata_GPU,5:Ny-4,1:4),view(odata_GPU_NSWE_tmp.odata_GPU_W_D2,5:Ny-4,1:4));
    copyto!(view(odata_GPU,5:Ny-4,Nx-3:Nx),view(odata_GPU_NSWE_tmp.odata_GPU_E_D2,5:Ny-4,1:4));

    copyto!(view(odata_GPU,1:1,1:Nx),view(odata_GPU,1,1:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_N_P,1,1:Nx));
    copyto!(view(odata_GPU,Ny,1:Nx),view(odata_GPU,Ny,1:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_S_P,4,1:Nx));
    copyto!(view(odata_GPU,1:Ny,1:4),view(odata_GPU,1:Ny,1:4).+view(odata_GPU_NSWE_tmp.odata_GPU_W_P,1:Ny,1:4));
    copyto!(view(odata_GPU,1:Ny,Nx-3:Nx),view(odata_GPU,1:Ny,Nx-3:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_E_P,1:Ny,1:4));

end

struct grid_info
    griddim_1D::Int64
    blockdim_1D::Int64
    griddim_2D::Tuple{Int64, Int64}
    blockdim_2D::Tuple{Int64, Int64}
end

function matrix_free_HA_GPU_v4(idata_GPU,odata_GPU,odata_GPU_NSWE_tmp,grid_info_GPU,coef_D,Nx,Ny,hx,hy)
    idata_GPU_N = @view idata_GPU[1:6,1:end]
    idata_GPU_S = @view idata_GPU[end-5:end,1:end]
    idata_GPU_W = @view idata_GPU[1:end,1:6]
    idata_GPU_E = @view idata_GPU[1:end,end-5:end]


    # odata_GPU_N_D2 = CuArray(zeros(4,Nx))
    # odata_GPU_S_D2 = CuArray(zeros(4,Nx))
    # odata_GPU_W_D2 = CuArray(zeros(Ny,4))
    # odata_GPU_E_D2 = CuArray(zeros(Ny,4))
    
    # odata_GPU_N_P = CuArray(zeros(4,Nx))
    # odata_GPU_S_P = CuArray(zeros(4,Nx))
    # odata_GPU_W_P = CuArray(zeros(Ny,4))
    # odata_GPU_E_P = CuArray(zeros(Ny,4))

    # Grid Information for GPU kernels
    # (Nx,Ny) = size(idata_GPU)
    # hx=hy = 1/(Nx-1)
    # TILE_DIM_1 = 16
    # TILE_DIM_2 = 16
    # griddim_2D = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	# blockdim_2D = (TILE_DIM_1,TILE_DIM_2)
    
    # TILE_DIM = 256
    # griddim_1D = div(Nx+TILE_DIM-1,TILE_DIM)
    # blockdim_1D = TILE_DIM

    # coeff_d = cudaconvert(CuArray([-1/12 4/3 -5/2 4/3 -1/12]))
    griddim_1D = grid_info_GPU.griddim_1D
    blockdim_1D = grid_info_GPU.blockdim_1D
    griddim_2D = grid_info_GPU.griddim_2D
    blockdim_2D= grid_info_GPU.blockdim_2D


    @cuda threads=blockdim_2D blocks=griddim_2D matrix_free_D2_p4_GPU_kernel_v3(idata_GPU,odata_GPU,Nx,Ny,hx,hy,Val(TILE_DIM_1),Val(TILE_DIM_2)); # kernel_v2 for - H_tilde * D2

    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_N_D2_kernel_(idata_GPU_N,odata_GPU_NSWE_tmp.odata_GPU_N_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_S_D2_kernel_(idata_GPU_S,odata_GPU_NSWE_tmp.odata_GPU_S_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_W_D2_kernel_(idata_GPU_W,odata_GPU_NSWE_tmp.odata_GPU_W_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_E_D2_kernel_(idata_GPU_E,odata_GPU_NSWE_tmp.odata_GPU_E_D2,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));

    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_N_P_kernel_(idata_GPU_N,odata_GPU_NSWE_tmp.odata_GPU_N_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_S_P_kernel_(idata_GPU_S,odata_GPU_NSWE_tmp.odata_GPU_S_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_W_P_kernel_(idata_GPU_W,odata_GPU_NSWE_tmp.odata_GPU_W_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));
    @cuda threads=blockdim_1D blocks=griddim_1D _matrix_free_E_P_kernel_(idata_GPU_E,odata_GPU_NSWE_tmp.odata_GPU_E_P,coef_D,Nx,Ny,hx,hy,Val(TILE_DIM));

    copyto!(view(odata_GPU,1:4,1:Nx),odata_GPU_NSWE_tmp.odata_GPU_N_D2);
    copyto!(view(odata_GPU,Ny-3:Ny,1:Nx),odata_GPU_NSWE_tmp.odata_GPU_S_D2);
    copyto!(view(odata_GPU,5:Ny-4,1:4),view(odata_GPU_NSWE_tmp.odata_GPU_W_D2,5:Ny-4,1:4));
    copyto!(view(odata_GPU,5:Ny-4,Nx-3:Nx),view(odata_GPU_NSWE_tmp.odata_GPU_E_D2,5:Ny-4,1:4));

    copyto!(view(odata_GPU,1:1,1:Nx),view(odata_GPU,1,1:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_N_P,1,1:Nx));
    copyto!(view(odata_GPU,Ny,1:Nx),view(odata_GPU,Ny,1:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_S_P,4,1:Nx));
    copyto!(view(odata_GPU,1:Ny,1:4),view(odata_GPU,1:Ny,1:4).+view(odata_GPU_NSWE_tmp.odata_GPU_W_P,1:Ny,1:4));
    copyto!(view(odata_GPU,1:Ny,Nx-3:Nx),view(odata_GPU,1:Ny,Nx-3:Nx).+view(odata_GPU_NSWE_tmp.odata_GPU_E_P,1:Ny,1:4));

end




struct MyStruct
    a :: Union{CuArray{Float32}, CuDeviceArray{Float32}}
end
