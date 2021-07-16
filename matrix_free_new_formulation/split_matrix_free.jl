using SparseArrays
using CUDA
using Random

function D2_split(idata,odata,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
        odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) / 2
    end 

    nothing
end



function matrix_free_cpu(idata,odata,Nx,Ny,h)
    (Nx,Ny) = size(idata)
    # odata = spzeros(Nx,Ny)
    idx = 1:3
    odata[idx,:] .= idata[idx,:]
    odata[Nx+1 .- idx,:] .= idata[Nx+1 .- idx,:]
    odata
    odata[:,idx] .= idata[:,idx]
    odata[:,Ny+1 .- idx] .= idata[:,Ny+1 .- idx]
    odata

    alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
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
    for i in 1:Nx
        for j = 1:Ny
            if !(3<= i <= Nx-3) && (3 <= j <= Ny-3)
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
    (i,j) = (1,1)
    odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    (i,j) = (1,Ny)
    odata[i,j] = (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    (i,j) = (Nx,1)
    odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+1]) / 4
    (i,j) = (Nx,Ny)
    odata[i,j] = (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-1]) / 4
    (i,j) = (1,2:Ny-1)
    odata[i,j] .= (idata[i,j] .- 2*idata[i+1,j] .+ idata[i+2,j] .+ idata[i,j .- 1] .- 2*idata[i,j] .+ idata[i,j .+ 1]) / 4
    (i,j) = (Nx,2:Ny-1)
    odata[i,j] .= (idata[i,j] .- 2*idata[i-1,j] .+ idata[i-2,j] .+ idata[i,j .- 1] .- 2*idata[i,j] .+ idata[i,j .+ 1]) / 4
    (i,j) = (2:Nx-1,1)
    odata[i,j] .= (idata[i.-1,j] .- 2*idata[i,j] .+ idata[i.+1,j] .+ idata[i,j] .- 2*idata[i,j+1] .+ idata[i,j+2]) / 4
    (i,j) = (2:Nx-1,Ny)
    odata[i,j] .= (idata[i.-1,j] .- 2*idata[i,j] .+ idata[i.+1,j] .+ idata[i,j] .- 2*idata[i,j-1] .+ idata[i,j - 2]) / 4
end

function test_matrix_free_cpu(level)
    Nx = Ny = 2^level+1
    h = 1/(Nx-1)
    Random.seed!(0)
    A = randn(Nx,Ny)
    odata = spzeros(Nx,Ny)
    matrix_free_cpu_v2(A,odata,Nx,Ny,h)
    t_cpu = time()
    iter_times = 100
    for i in 1:iter_times
        matrix_free_cpu_v2(A,odata,Nx,Ny,h)
    end
    t_cpu = time() - t_cpu
    @show t_cpu


    t_convert = time()
    for i in 1:iter_times
        cu_out = CUDA.CUSPARSE.CuSparseMatrixCSC(odata)
    end
    synchronize()
    t_convert = time() - t_convert
    @show t_convert
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

    @cuda threads=blockdim blocks=griddim D2_split(cu_A,cu_out,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))

    iter_times = 100

    t_D2_start = time()
    for _ in 1:iter_times
        @cuda threads=blockdim blocks=griddim D2_split(cu_A,cu_out,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    end
    synchronize()
    t_D2 = time() - t_D2_start

    @show t_D2
end