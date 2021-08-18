using SparseArrays 
using CUDA
using Random
using Adapt
using LinearAlgebra

function D2_split_naive(idata,odata,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    @inbounds if 0 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = 0 # maybe do this on local memory
    end

    @inbounds if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
        odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
    end 

    nothing
end

function D2_split(idata,odata,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    si = tidx + 1
    sj = tidy + 1


    HALO_WIDTH = 2 # For second order derivative

    tile = @cuStaticSharedMem(eltype(idata), (TILE_DIM1+2, TILE_DIM2+2))

    global_index = (i-1)*Ny+j
    @inbounds if 0 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = 0 # maybe do this on local memory
    end


    @inbounds if 1 <= i <= Nx && 1 <= j <= Ny
        tile[si,sj] = idata[i,j]
    end
    sync_threads()

    # if tidx == 1 && 1 <= j <= Ny
    #     tile[si-1,sj] = tile[Nx+si-2,sj]
    # #     tile[si+Nx,sj] = tile[si,sj]
    # end

    # sync_threads()

    # if 1 <= i <= Nx && j <= 1
    #     tile[si,sj-1] = tile[si,Ny+sj-2]
    #     tile[si,sj+Ny] = tile[si,sj]
    # end

    sync_threads()

    # @inbounds if 2 <= i <= Nx -1 && 2 <= j <= Ny - 1 &&  1 <= si <= TILE_DIM1 + 1 && 1 <= sj <= TILE_DIM2 + 1 
    #     odata[i,j] = (tile[si-1,sj] + tile[si+1,sj] + tile[si,sj-1] + tile[si,sj+1] - 4*tile[si,sj])
    # end 

    nothing
end


struct boundary_containers
    CPU_W
    CPU_W_T
    CPU_E
    CPU_E_T
    CPU_N
    CPU_S
    CPU_OUT_W
    CPU_OUT_E
    CPU_OUT_N
    CPU_OUT_S
end

function matrix_free_A(idata,odata)
    # odata .= 0
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

    # CPU_W = zeros(Nx,3)
    # CPU_W_T = zeros(3,Nx)
    # CPU_E = zeros(Nx,3)
    # CPU_E_T = zeros(3,Nx)
    # CPU_N = zeros(3,Ny)
    # CPU_S = zeros(3,Ny)


    # CPU_OUT_W = zeros(Nx,3)
    # CPU_OUT_E = zeros(Nx,3)
    # CPU_OUT_N = zeros(3,Ny)
    # CPU_OUT_S = zeros(3,Ny)

    CPU_W = Array{Float64,2}(undef,Nx,3)
    CPU_W_T = Array{Float64,2}(undef,3,Nx)
    CPU_E = Array{Float64,2}(undef,Nx,3)
    CPU_E_T = Array{Float64,2}(undef,3,Nx)
    CPU_N = Array{Float64,2}(undef,3,Ny)
    CPU_S = Array{Float64,2}(undef,3,Ny)


    

    CPU_OUT_W = Array{Float64,2}(undef,Nx,3)
    CPU_OUT_E = Array{Float64,2}(undef,Nx,3)
    CPU_OUT_N = Array{Float64,2}(undef,3,Ny)
    CPU_OUT_S = Array{Float64,2}(undef,3,Ny)

    # copyto!(CPU_W,idata[:,1:3])
    # copyto!(CPU_E,idata[:,end-2:end])
    # copyto!(CPU_N,idata[1:3,:])
    # copyto!(CPU_S,idata[end-2:end,:])

    copyto!(CPU_W,view(idata,:,1:3))
    copyto!(CPU_E,view(idata,:,Ny-2:Ny))
    copyto!(CPU_N,view(idata,1:3,:))
    copyto!(CPU_S,view(idata,Nx-2:Nx,:))

    @cuda threads=blockdim blocks=griddim D2_split_naive(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))

    ## CPU Calculation

    CPU_W_T .= CPU_W'
    CPU_E_T .= CPU_E'
    # CPU_OUT_W_T = zeros(3,Ny)
    # CPU_OUT_E_T = zeros(3,Ny)
    CPU_OUT_W_T = Array{Float64,2}(undef,3,Ny)
    CPU_OUT_E_T = Array{Float64,2}(undef,3,Ny)


    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1

    i = 1
    # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        CPU_OUT_N[1,j] += (CPU_N[1,j] - 2*CPU_N[2,j] + CPU_N[3,j] + CPU_N[1,j-1] - 2* CPU_N[1,j] + CPU_N[1,j+1] + 2 * alpha3 * (1.5 * CPU_N[1,j] - 2*CPU_N[2,j] + 0.5*CPU_N[3,j])) / 2
    end
    synchronize()

    i = Nx
    # # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        CPU_OUT_S[3,j] += (CPU_S[3,j] - 2*CPU_S[2,j] + CPU_S[1,j] + CPU_S[3,j-1] - 2* CPU_S[3,j] + CPU_S[3,j+1] + 2 * alpha4 * (1.5 * CPU_S[3,j] - 2*CPU_S[2,j] + 0.5*CPU_S[1,j])) / 2
    end
    # synchronize()

    j = 1

    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        CPU_OUT_W_T[1,i] += (CPU_W_T[1,i-1] - 2*CPU_W_T[1,i] + CPU_W_T[1,i+1] + CPU_W_T[1,i] - 2*CPU_W_T[2,i] + CPU_W_T[3,i]) / 2
        CPU_OUT_W_T[1,i] += (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        CPU_OUT_W_T[2,i] += (2 * beta * (-1 * CPU_W_T[1,i]))
        CPU_OUT_W_T[3,i] += (0.5 * beta * CPU_W_T[1,i])
    end

    # # odata[:,1] .+= odata_W_T[1,:]

   

    j = Ny
    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        CPU_OUT_E_T[3,i] += (CPU_E_T[3,i-1] - 2*CPU_E_T[3,i] + CPU_E_T[3,i+1] + CPU_E_T[3,i] - 2*CPU_E_T[2,i] + CPU_E_T[1,i]) / 2
        CPU_OUT_E_T[3,i] += (2 * beta * (1.5 * CPU_E_T[3,i]) + 2 * alpha1 * CPU_E_T[3,i] * h) / 2
        CPU_OUT_E_T[2,i] += (2 * beta * (-1 * CPU_E_T[3,i]))
        CPU_OUT_E_T[1,i] += (0.5 * beta * CPU_E_T[3,i])
    end

    (i,j) = (1,1)


    CPU_OUT_N[1,j] += (CPU_W[i,j] - 2*CPU_W[i+1,j] + CPU_W[i+2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    CPU_OUT_N[1,j] += 2 * alpha3 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i+1,j] + 0.5*CPU_W[i+2,j])) / 4 # Neumann

    CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    CPU_OUT_N[1,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    CPU_OUT_N[1,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet


    (i,j) = (1,Ny)
    CPU_OUT_N[1,j] += (CPU_E[i,3] - 2*CPU_E[i+1,3] + CPU_E[i+2,3] + CPU_E[i,3] - 2*CPU_E[i,2] + CPU_E[i,1]) / 4 # D2
    
    CPU_OUT_N[1,j] += 2 * alpha3 * (1.5 * CPU_E[i,3] - 2*CPU_E[i+1,3] + 0.5 * CPU_E[i+2,3]) / 4 # Neumann
    CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_E[i,3]) + 2 * alpha2 * (CPU_E[i,3]) * h) / 4 # Dirichlet
    CPU_OUT_N[1,j-1] += (2 * beta * (-1 * CPU_E[i,3])) / 2 # Dirichlet
    CPU_OUT_N[1,j-2] += (0.5 * beta * (CPU_E[i,3])) / 2 # Dirichlet



    (i,j) = (Nx,1)
    CPU_OUT_S[3,j] += (CPU_W[i,j] - 2*CPU_W[i-1,j] + CPU_W[i-2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    CPU_OUT_S[3,j] += 2 * alpha4 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i-1,j] + 0.5*CPU_W[i-2,j])) / 4 # Neumann
    CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    CPU_OUT_S[3,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    CPU_OUT_S[3,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet

    (i,j) = (Nx,Ny)
    CPU_OUT_S[3,j] += (CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + CPU_E[Nx-2,3] + CPU_E[Nx,3] - 2*CPU_E[Nx,2] + CPU_E[Nx,1]) / 4 # D2

    CPU_OUT_S[3,j] += 2 * alpha4 * (1.5 * CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + 0.5 * CPU_E[Nx-2,3]) / 4 # Neumann
    CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_E[Nx,3]) + 2 * alpha2 * (CPU_E[Nx,3]) * h) / 4 # Dirichlet
    CPU_OUT_S[3,j-1] += (2 * beta * (-1 * CPU_E[Nx,3])) / 2 # Dirichlet
    CPU_OUT_S[3,j-2] += (0.5 * beta * (CPU_E[Nx,3])) / 2 # Dirichlet

    ## End CPU calculation

    synchronize()

    # Copy W & E boundary
    copyto!(view(odata,1:Nx,1:3),view(odata,1:Nx,1:3) .+ CuArray(CPU_OUT_W_T)')
    copyto!(view(odata,1:Nx,Ny-2:Ny),view(odata,1:Nx,Ny-2:Ny) .+ CuArray(CPU_OUT_E_T)')

    # Copy N & S boundary
    copyto!(view(odata,1,1:Ny),view(odata,1,1:Ny) .+ CuArray(CPU_OUT_N[1,:]))
    copyto!(view(odata,Nx,1:Ny),view(odata,Nx,1:Ny) .+ CuArray(CPU_OUT_S[end,:]))
    nothing
end



function test_matrix_free_A(level)
    Nx = Ny = 2^level + 1
    h = 1/(Nx-1)
    println("")
    println("Starting Test")
    println("2D Domain Size: $Nx by $Ny")
    Random.seed!(0)
    idata = CuArray(randn(Nx,Ny))
    odata = CUDA.zeros(Nx,Ny)

    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim D2_split(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    @cuda threads=blockdim blocks=griddim D2_split_naive(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))

    boundary_data = boundary_containers(zeros(Nx,3),zeros(3,Nx),zeros(Nx,3),zeros(3,Nx),zeros(3,Ny),zeros(3,Ny),zeros(Nx,3),zeros(Nx,3),zeros(3,Ny),zeros(3,Ny))
    matrix_free_A(idata,odata)


    iter_times = 1000
    # Evaluating only D2
    t_start_D2 = time()
    for _ in 1:iter_times
        @cuda threads=blockdim blocks=griddim D2_split(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    end
    synchronize()
    t_D2 = (time() - t_start_D2) * 1000 / iter_times
    @show t_D2
    # End evaluating D2

    # Evaluating only D2_naive
    t_start_D2_naive = time()
    for _ in 1:iter_times
        @cuda threads=blockdim blocks=griddim D2_split_naive(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    end
    synchronize()
    t_D2_naive = (time() - t_start_D2_naive) * 1000 / iter_times
    @show t_D2_naive
    # End evaluating D2_naive

    t_start_A = time()
    for _ in 1:iter_times
         matrix_free_A(idata,odata)
    end
    synchronize()
    t_A = (time() - t_start_A) * 1000 / iter_times
    @show t_A 

    through_put = sizeof(idata) * 1e-6 / t_D2
    @show through_put

end

function CG_GPU(b_reshaped_GPU,x_GPU)
    (Nx,Ny) = size(b_reshaped_GPU)
    odata = CUDA.zeros(Nx,Ny)
    # odata_D2_GPU = CUDA.zeros(Nx,Ny)
    # odata_boundary_GPU = CUDA.zeros(Nx,Ny)
    # matrix_free_A_v3(x_GPU,odata,odata_D2_GPU,odata_boundary_GPU)
    matrix_free_A_v4(x_GPU,odata)
    r_GPU = b_reshaped_GPU - odata
    p_GPU = copy(r_GPU)
    rsold_GPU = sum(r_GPU .* r_GPU)
    Ap_GPU = CUDA.zeros(Nx,Ny)
    num_iter_steps = 0
    for i in 1:Nx*Ny
    # for i in 1:2071
        num_iter_steps += 1
        # @show i
        # @show rsold_GPU
        # matrix_free_A_v3(p_GPU,Ap_GPU,odata_D2_GPU,odata_boundary_GPU)
        matrix_free_A_v4(p_GPU,Ap_GPU)
        alpha_GPU = rsold_GPU / (sum(p_GPU .* Ap_GPU))
        # x_GPU .= x_GPU .+ alpha_GPU * p_GPU
        x_GPU .+= alpha_GPU * p_GPU
        # r_GPU .= r_GPU .- alpha_GPU * Ap_GPU
        r_GPU .-= alpha_GPU * Ap_GPU
        # CUDA.CUBLAS.axpy!()
        rsnew_GPU = sum(r_GPU .* r_GPU)
        if sqrt(rsnew_GPU) < 1e-10
            break
        end
        p_GPU .= r_GPU .+ (rsnew_GPU/rsold_GPU) * p_GPU
        rsold_GPU = rsnew_GPU
    end
    @show num_iter_steps
end

function CG_CPU(A,b,x)
    r = b - A * x;
    p = r;
    rsold = r' * r
    # Ap = spzeros(length(b))
    Ap = similar(b)

    for i = 1:length(b)
    # for i = 1:2
        # @show i
        # @show rsold
        # Ap = A * p;
        mul!(Ap,A,p)
        alpha = rsold / (p' * Ap)
        x .= x .+ alpha * p;
        r .= r .- alpha * Ap;
        rsnew = r' * r
        if sqrt(rsnew) < 1e-10
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew
    end
end