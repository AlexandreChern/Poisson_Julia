# function matrix_free_cpu_old(idata,odata,Nx,Ny,h)
#     (Nx,Ny) = size(idata)
#     # odata = spzeros(Nx,Ny)
#     idx = 1:3
#     odata[idx,:] .= idata[idx,:]
#     odata[Nx+1 .- idx,:] .= idata[Nx+1 .- idx,:]
#     odata
#     odata[:,idx] .= idata[:,idx]
#     odata[:,Ny+1 .- idx] .= idata[:,Ny+1 .- idx]
#     odata

#     alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
#     offset_j = spzeros(Int,Ny)
#     offset_j[2] = -1
#     offset_j[3] = -2
#     offset_j[Ny-1] = 1
#     offset_j[Ny-2] = 2
#     coef_j = spzeros(Ny)
#     coef_j[1] = 2* 1.5
#     coef_j[2] = - 2.0
#     coef_j[3] = 0.5
#     coef_j[Ny] = 2*1.5
#     coef_j[Ny-1] = -2.0
#     coef_j[Ny-2] = 0.5
#     for i in 1:Nx
#         for j = 1:Ny
#             if !(3<= i <= Nx-3) && (3 <= j <= Ny-3)
#                 global_index = (i-1)*Ny+j

#                 offset_x = div(2*Nx-i-3,Nx-2) - 1
#                 offset_y = div(2*Ny-j-3,Ny-2) - 1

#                 idata_index_x = (i-1)*Ny + j + (offset_x) * Nx
#                 idata_index_y = (i-1)*Ny + j + offset_y

#                 odata[global_index] = (( (idata[idata_index_x-Ny] - 2*idata[idata_index_x] + idata[idata_index_x + Ny])  + (idata[idata_index_y-1] - 2*idata[idata_index_y] + idata[idata_index_y + 1]) )   
#                 + abs(offset_y) * 2 * alpha1 * ( (1.5*idata[global_index]) - 2*idata[global_index+offset_y] + 0.5*idata[global_index+2*offset_y])
#                 + abs(offset_x) * alpha4 * 2 * (idata[global_index] * h) 
#                 + coef_j[i] * beta * (idata[global_index+offset_j[i] * Ny])
#                 ) / 2^(abs(offset_x) + abs(offset_y))
#             end
#         end
#     end
# end

function Boundary_GPU(idata,odata,Nx,Ny,h::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1,TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = (i-1)*Ny+j

    idata_N = idata[]
    idata_S = 
    idata_W = 
    idata_E =

    nothing
end

# function matrix_free_cpu_v5(GPU_Array,odata,Nx,Ny,h)
#     # input as idata_gpu
#     # pure GPU kernel

#     odata .= 0
    

#     # alpha1 = alpha2 = alpha3 = alpha4 = beta = 1
#     alpha1 = alpha2 = -13/h
#     alpha3 = alpha4 = -1
#     beta = 1
#     (i,j) = (1,1)


#     odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

#     odata[i,j] += 2 * alpha3 * (( 1.5* idata[i,j] - 2*idata[i+1,j] + 0.5*idata[i+2,j])) / 4 # Neumann

#     odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
#     odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
#     odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


#     (i,j) = (1,Ny)
#     odata[i,j] += (idata[i,j] - 2*idata[i+1,j] + idata[i+2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2
    
#     odata[i,j] += 2 * alpha3 * (1.5 * idata[i,j] - 2*idata[i+1,j] + 0.5 * idata[i+2,j]) / 4 # Neumann
#     odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
#     odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
#     odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet



#     (i,j) = (Nx,1)
#     odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j+1] + idata[i,j+2]) / 4 # D2

#     odata[i,j] += 2 * alpha4 * (( 1.5* idata[i,j] - 2*idata[i-1,j] + 0.5*idata[i-2,j])) / 4 # Neumann
#     odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha1 * (idata[i,j]) * h) / 4 # Dirichlet
#     odata[i,j+1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
#     odata[i,j+2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet

#     (i,j) = (Nx,Ny)
#     odata[i,j] += (idata[i,j] - 2*idata[i-1,j] + idata[i-2,j] + idata[i,j] - 2*idata[i,j-1] + idata[i,j-2]) / 4 # D2

#     odata[i,j] += 2 * alpha4 * (1.5 * idata[i,j] - 2*idata[i-1,j] + 0.5 * idata[i-2,j]) / 4 # Neumann
#     odata[i,j] += (2 * beta * (1.5 * idata[i,j]) + 2 * alpha2 * (idata[i,j]) * h) / 4 # Dirichlet
#     odata[i,j-1] += (2 * beta * (-1 * idata[i,j])) / 2 # Dirichlet
#     odata[i,j-2] += (0.5 * beta * (idata[i,j])) / 2 # Dirichlet


#     # (i,j) = (1,2:Ny-1)
#     i = 1
#     idata_N = view(idata,1:3,1:Ny)
#     # Threads.@threads for j in 2:Ny-1
#     @inbounds for j in 2:Ny-1
#         odata[1,j] += (idata_N[1,j] - 2*idata_N[2,j] + idata_N[3,j] + idata_N[1,j-1] - 2* idata_N[1,j] + idata_N[1,j+1] + 2 * alpha3 * (1.5 * idata_N[1,j] - 2*idata_N[2,j] + 0.5*idata_N[3,j])) / 2
#     end
#     # synchronize()

#     i = Nx
#     idata_S = view(idata,Nx-2:Nx,1:Ny)
#     # Threads.@threads for j in 2:Ny-1
#     @inbounds for j in 2:Ny-1
#         odata[i,j] += (idata_S[3,j] - 2*idata_S[2,j] + idata_S[1,j] + idata_S[3,j-1] - 2* idata_S[3,j] + idata_S[3,j+1] + 2 * alpha4 * (1.5 * idata_S[3,j] - 2*idata_S[2,j] + 0.5*idata_S[1,j])) / 2
#     end
#     # synchronize()

#     j = 1
#     idata_W = view(idata,1:Nx,1:3)

#     # @inbounds for i in 2:Nx-1
#     #     odata[i,j] += (idata_W[i-1,1] - 2*idata_W[i,1] + idata_W[i+1,1] + idata_W[i,1] - 2*idata_W[i,2] + idata_W[i,3]) / 2
#     #     # odata[i,j] += (2 * beta * (1.5 * idata_W[i,1]) + 2 * alpha2 * idata_W[i,1] * h) / 2
#     #     # odata[i,j+1] += (2 * beta * (-1 * idata_W[i,1]))
#     #     # odata[i,j+2] += (0.5 * beta * idata_W[i,1])
#     # end

#     CPU_W_T = copy(idata_W')
#     odata_W_T= zeros(size(CPU_W_T))
#     # @inbounds for i in 2:Nx-1
#     @inbounds for i in 2:Nx-1
#         odata_W_T[1,i] += (CPU_W_T[1,i-1] - 2*CPU_W_T[1,i] + CPU_W_T[1,i+1] + CPU_W_T[1,i] - 2*CPU_W_T[2,i] + CPU_W_T[3,i]) / 2
#         odata_W_T[1,i] += (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
#         odata_W_T[2,i] += (2 * beta * (-1 * CPU_W_T[1,i]))
#         odata_W_T[3,i] += (0.5 * beta * CPU_W_T[1,i])
#     end

#     # odata[:,1] .+= odata_W_T[1,:]

   

#     j = Ny
#     idata_E = view(idata,1:Nx,Ny-2:Ny)
#     CPU_E_T = copy(idata_E')
#     CPU_OUT_E_T = zeros(size(CPU_E_T))
#     @inbounds for i in 2:Nx-1
#         CPU_OUT_E_T[3,i] += (CPU_E_T[3,i-1] - 2*CPU_E_T[3,i] + CPU_E_T[3,i+1] + CPU_E_T[3,i] - 2*CPU_E_T[2,i] + CPU_E_T[1,i]) / 2
#         CPU_OUT_E_T[3,i] += (2 * beta * (1.5 * CPU_E_T[3,i]) + 2 * alpha1 * CPU_E_T[3,i] * h) / 2
#         CPU_OUT_E_T[2,i] += (2 * beta * (-1 * CPU_E_T[3,i]))
#         CPU_OUT_E_T[1,i] += (0.5 * beta * CPU_E_T[3,i])
#     end

#     # @inbounds for i in 2:Nx-1
#     #     odata[i,j] += (idata_E[i-1,3] - 2*idata_E[i,3] + idata_E[i+1,3] + idata_E[i,3] - 2*idata_E[i,2] + idata_E[i,1]) / 2
#     #     odata[i,j] += (2 * beta * (1.5 * idata_E[i,3]) + 2 * alpha1 * idata_E[i,3] * h) / 2
#     #     odata[i,j-1] += (2 * beta * (-1 * idata_E[i,3]))
#     #     odata[i,j-2] += (0.5 * beta * idata_E[i,3])
#     # end

#     odata[:,1:3] .+= odata_W_T'
#     odata[:,end-2:end] .+= CPU_OUT_E_T'
# end


function test_copy_data(level)
    Nx = Ny = 2^level + 1
    h = 1/(Nx-1)
    println("")
    println("Starting Test")
    println("2D Domain Size: $Nx by $Ny")
    Random.seed!(0)
    # idata = sparse(randn(Nx,Ny))
    idata = CuArray(randn(Nx,Ny))
    # odata = spzeros(Nx,Ny)
    odata = CuArray(randn(Nx,Ny))
    println("Size of the solution matrix (GPU): ", sizeof(idata), " Bytes")

    idata_cpu = zeros(Nx,Ny)
    odata_cpu = spzeros(Nx,Ny)
    copyto!(idata_cpu,idata)
    println("Size of the solution matrix (CPU): ", sizeof(idata_cpu), " Bytes")


    GPU_Array = CuArray(randn(Nx,Ny))

    CPU_Array = zeros(Nx,Ny)
    
    CPU_W = zeros(Nx,3)
    CPU_E = zeros(Nx,3)

    CPU_N = zeros(3,Ny)
    CPU_S = zeros(3,Ny)

    t_copy_dense = time()
    iter_times_copy_data = 40
    for _ in 1:iter_times_copy_data
        copyto!(CPU_Array,GPU_Array)
    end
    t_copy_dense = ( time() - t_copy_dense ) * 1000 / iter_times_copy_data
    @show t_copy_dense

    t_copy_sparse = time()
    iter_times_copy_data = 40
    for _ in 1:iter_times_copy_data
        copyto!(CPU_W,GPU_Array[:,1:3])
        copyto!(CPU_E,GPU_Array[:,end-2:end])
        copyto!(CPU_N,GPU_Array[1:3,:])
        copyto!(CPU_S,GPU_Array[end-2:end,:])
    end
    t_copy_sparse = ( time() - t_copy_sparse ) * 1000 / iter_times_copy_data
    @show t_copy_sparse
end