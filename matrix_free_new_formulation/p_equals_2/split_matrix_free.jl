using SparseArrays 
using KernelAbstractions
using KernelAbstractions.Extras
using CUDA
using Random
using Adapt
using LinearAlgebra
using Printf

function D2_split_naive(idata,odata,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    if 1 <= i <= Nx && 1 <= j <= Ny
        odata[i,j] = 0 # maybe do this on local memory
        # odata[i,j] = 0 # maybe do this on local memory
    end
    sync_threads()

    if 2 <= i <= Nx-1 && 2 <= j <= Ny - 1
        @inbounds   odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
        # odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
    end 

    nothing
end


function D2_split_naive_v2(idata,odata,Nx,Ny,h,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    if i <= Nx && j <= Ny
        if i == 1 || i == Nx || j == 1 || j == Ny
            @inbounds odata[i,j] = 0
        else
            @inbounds  odata[i,j] = (idata[i-1,j] + idata[i+1,j] + idata[i,j-1] + idata[i,j+1] - 4*idata[i,j]) 
       end
    #    @inbounds odata[i,j] = idata[i,j]
    end

    nothing
end

function copy_naive!(b, a)
    N = size(a, 1)

    # which thread are we in our block
    tidx = threadIdx().x
    tidy = threadIdx().y

    # which block of threads are we in
    bidx = blockIdx().x
    bidy = blockIdx().y

    # What is the size of the thread block
    dimx = blockDim().x
    dimy = blockDim().y

    # what index am I in the global thread space
    i = tidx + dimx * (bidx - 1)
    j = tidy + dimy * (bidy - 1)

    if i <= N && j <= N
        # aval = a[i, j]
        # b[i, j] = aval
        @inbounds b[i, j] = a[i, j]
    end

    return nothing
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
    if 0 <= i <= Nx && 1 <= j <= Ny
        @inbounds  odata[i,j] = 0 # maybe do this on local memory
    end


    if 1 <= i <= Nx && 1 <= j <= Ny
        @inbounds  tile[si,sj] = idata[i,j]
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

    if 2 <= i <= Nx -1 && 2 <= j <= Ny - 1 &&  1 <= si <= TILE_DIM1 + 1 && 1 <= sj <= TILE_DIM2 + 1 
        @inbounds odata[i,j] = (tile[si-1,sj] + tile[si+1,sj] + tile[si,sj-1] + tile[si,sj+1] - 4*tile[si,sj])
    end 

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



    CPU_W = Array{Float64,2}(undef,Nx,3)
    CPU_W_T = Array{Float64,2}(undef,3,Nx)
    CPU_E = Array{Float64,2}(undef,Nx,3)
    CPU_E_T = Array{Float64,2}(undef,3,Nx)   
    CPU_N = Array{Float64,2}(undef,3,Ny)
    CPU_S = Array{Float64,2}(undef,3,Ny)

    # CPU_OUT_W = Array{Float64,2}(undef,Nx,3)
    # CPU_OUT_W_T = Array{Float64,2}(undef,3,Nx)
    # CPU_OUT_E = Array{Float64,2}(undef,Nx,3)
    # CPU_OUT_E_T = Array{Float64,2}(undef,3,Nx)
    # CPU_OUT_N = Array{Float64,2}(undef,3,Ny)
    # CPU_OUT_S = Array{Float64,2}(undef,3,Ny)

    copyto!(CPU_W,view(idata,:,1:3))
    copyto!(CPU_E,view(idata,:,Ny-2:Ny))
    copyto!(CPU_N,view(idata,1:3,:))
    copyto!(CPU_S,view(idata,Nx-2:Nx,:))

    @cuda threads=blockdim blocks=griddim D2_split_naive(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))

    
    CPU_OUT_W = zeros(Nx,3)
    CPU_OUT_W_T = zeros(3,Nx)
    CPU_OUT_E = zeros(Nx,3)
    CPU_OUT_E_T = zeros(3,Nx)
    CPU_OUT_N = zeros(3,Ny)
    CPU_OUT_S = zeros(3,Ny)
    
    
    # CPU_OUT_W = Array{Float64,2}(undef,Nx,3)
    # CPU_OUT_W_T = Array{Float64,2}(undef,3,Nx)
    # CPU_OUT_E = Array{Float64,2}(undef,Nx,3)
    # CPU_OUT_E_T = Array{Float64,2}(undef,3,Nx)
    # CPU_OUT_N = Array{Float64,2}(undef,3,Ny)
    # CPU_OUT_S = Array{Float64,2}(undef,3,Ny)

    

    ## CPU Calculation

    CPU_W_T .= CPU_W'
    CPU_E_T .= CPU_E'
    # CPU_OUT_W_T = Array{Float64,2}(undef,3,Ny)
    # CPU_OUT_E_T = Array{Float64,2}(undef,3,Ny)


    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1

    i = 1
    # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        @inbounds CPU_OUT_N[1,j] += (CPU_N[1,j] - 2*CPU_N[2,j] + CPU_N[3,j] + CPU_N[1,j-1] - 2* CPU_N[1,j] + CPU_N[1,j+1] + 2 * alpha3 * (1.5 * CPU_N[1,j] - 2*CPU_N[2,j] + 0.5*CPU_N[3,j])) / 2
    end
    synchronize()

    i = Nx
    # # Threads.@threads for j in 2:Ny-1
    # @inbounds for j in 2:Ny-1
    for j in 2:Ny-1
        @inbounds CPU_OUT_S[3,j] += (CPU_S[3,j] - 2*CPU_S[2,j] + CPU_S[1,j] + CPU_S[3,j-1] - 2* CPU_S[3,j] + CPU_S[3,j+1] + 2 * alpha4 * (1.5 * CPU_S[3,j] - 2*CPU_S[2,j] + 0.5*CPU_S[1,j])) / 2
    end
    # synchronize()

    j = 1

    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        @inbounds CPU_OUT_W_T[1,i] += (CPU_W_T[1,i-1] - 2*CPU_W_T[1,i] + CPU_W_T[1,i+1] + CPU_W_T[1,i] - 2*CPU_W_T[2,i] + CPU_W_T[3,i]) / 2
        @inbounds CPU_OUT_W_T[1,i] += (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2 # should be using alpha1 maybe
        @inbounds CPU_OUT_W_T[2,i] += (2 * beta * (-1 * CPU_W_T[1,i]))
        @inbounds CPU_OUT_W_T[3,i] += (0.5 * beta * CPU_W_T[1,i])
    end

    # # odata[:,1] .+= odata_W_T[1,:]

   

    j = Ny
    # @inbounds for i in 2:Nx-1
    for i in 2:Nx-1
        @inbounds CPU_OUT_E_T[3,i] += (CPU_E_T[3,i-1] - 2*CPU_E_T[3,i] + CPU_E_T[3,i+1] + CPU_E_T[3,i] - 2*CPU_E_T[2,i] + CPU_E_T[1,i]) / 2
        @inbounds CPU_OUT_E_T[3,i] += (2 * beta * (1.5 * CPU_E_T[3,i]) + 2 * alpha1 * CPU_E_T[3,i] * h) / 2
        @inbounds CPU_OUT_E_T[2,i] += (2 * beta * (-1 * CPU_E_T[3,i]))
        @inbounds CPU_OUT_E_T[1,i] += (0.5 * beta * CPU_E_T[3,i])
    end

    (i,j) = (1,1)


    @inbounds CPU_OUT_N[1,j] += (CPU_W[i,j] - 2*CPU_W[i+1,j] + CPU_W[i+2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    @inbounds CPU_OUT_N[1,j] += 2 * alpha3 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i+1,j] + 0.5*CPU_W[i+2,j])) / 4 # Neumann

    @inbounds CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    @inbounds CPU_OUT_N[1,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    @inbounds CPU_OUT_N[1,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet


    (i,j) = (1,Ny)
    @inbounds CPU_OUT_N[1,j] += (CPU_E[i,3] - 2*CPU_E[i+1,3] + CPU_E[i+2,3] + CPU_E[i,3] - 2*CPU_E[i,2] + CPU_E[i,1]) / 4 # D2
    
    @inbounds CPU_OUT_N[1,j] += 2 * alpha3 * (1.5 * CPU_E[i,3] - 2*CPU_E[i+1,3] + 0.5 * CPU_E[i+2,3]) / 4 # Neumann
    @inbounds CPU_OUT_N[1,j] += (2 * beta * (1.5 * CPU_E[i,3]) + 2 * alpha2 * (CPU_E[i,3]) * h) / 4 # Dirichlet
    @inbounds CPU_OUT_N[1,j-1] += (2 * beta * (-1 * CPU_E[i,3])) / 2 # Dirichlet
    @inbounds CPU_OUT_N[1,j-2] += (0.5 * beta * (CPU_E[i,3])) / 2 # Dirichlet



    (i,j) = (Nx,1)
    @inbounds CPU_OUT_S[3,j] += (CPU_W[i,j] - 2*CPU_W[i-1,j] + CPU_W[i-2,j] + CPU_W[i,j] - 2*CPU_W[i,j+1] + CPU_W[i,j+2]) / 4 # D2

    @inbounds CPU_OUT_S[3,j] += 2 * alpha4 * (( 1.5* CPU_W[i,j] - 2*CPU_W[i-1,j] + 0.5*CPU_W[i-2,j])) / 4 # Neumann
    @inbounds CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_W[i,j]) + 2 * alpha1 * (CPU_W[i,j]) * h) / 4 # Dirichlet
    @inbounds CPU_OUT_S[3,j+1] += (2 * beta * (-1 * CPU_W[i,j])) / 2 # Dirichlet
    @inbounds CPU_OUT_S[3,j+2] += (0.5 * beta * (CPU_W[i,j])) / 2 # Dirichlet

    (i,j) = (Nx,Ny)
    @inbounds CPU_OUT_S[3,j] += (CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + CPU_E[Nx-2,3] + CPU_E[Nx,3] - 2*CPU_E[Nx,2] + CPU_E[Nx,1]) / 4 # D2

    @inbounds CPU_OUT_S[3,j] += 2 * alpha4 * (1.5 * CPU_E[Nx,3] - 2*CPU_E[Nx-1,3] + 0.5 * CPU_E[Nx-2,3]) / 4 # Neumann
    @inbounds CPU_OUT_S[3,j] += (2 * beta * (1.5 * CPU_E[Nx,3]) + 2 * alpha2 * (CPU_E[Nx,3]) * h) / 4 # Dirichlet
    @inbounds CPU_OUT_S[3,j-1] += (2 * beta * (-1 * CPU_E[Nx,3])) / 2 # Dirichlet
    @inbounds CPU_OUT_S[3,j-2] += (0.5 * beta * (CPU_E[Nx,3])) / 2 # Dirichlet

    ## End CPU calculation

    @inbounds CPU_OUT_E .= CPU_OUT_E_T'
    @inbounds CPU_OUT_W .= CPU_OUT_W_T'

    synchronize()

    # @show CPU_OUT_N
    # @show CPU_OUT_S
    # @show CPU_OUT_W
    # @show CPU_OUT_E

    # show(stdout, "text/plain", odata)
    # println()

    # show(stdout, "text/plain", CPU_OUT_W)
    # println()
    # show(stdout, "text/plain", CPU_OUT_E)
    # println()
    # show(stdout, "text/plain", CPU_OUT_N)
    # println()
    # show(stdout, "text/plain", CPU_OUT_S)
    # println()

    ## Copy W & E boundary
    copyto!(view(odata,1:Nx,1:3),view(odata,1:Nx,1:3) + CuArray(CPU_OUT_W))
    copyto!(view(odata,1:Nx,Ny-2:Ny),view(odata,1:Nx,Ny-2:Ny) + CuArray(CPU_OUT_E))

    # Copy N & S boundary
    copyto!(view(odata,1,1:Ny),view(odata,1,1:Ny) + CuArray(CPU_OUT_N[1,:]))
    copyto!(view(odata,Nx,1:Ny),view(odata,Nx,1:Ny) + CuArray(CPU_OUT_S[end,:]))

    # show(stdout, "text/plain", odata)
    # println()
    synchronize()
    nothing
end

function SBP_N!(idata,odata,alpha1,alpha2,alpha3,beta,h)
    (Nx,Ny) = size(idata)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    if 4 <= idx <= Ny - 3
        odata[1,idx] = (idata[1,idx] - 2*idata[2,idx] + idata[3,idx] + idata[1,idx-1] - 2*idata[1,idx] + idata[1,idx+1] + 2 * alpha3 * (1.5 * idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx])) / 2
    end
    # sync_threads()
    if idx == 1
        odata[1,idx] = (idata[1,idx] - 2*idata[1,idx+1] + idata[1,idx+2] + idata[1,idx] - 2*idata[2,idx] + idata[3,idx] 
                        + 2 * alpha3 * (( 1.5* idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx])) ) / 4 
                        # + 2 * beta * (1.5 * idata[1,idx]) + 2 * alpha1 * (idata[1,idx]) * h
                        
        odata[1,idx+1] =  (idata[1,idx+1] - 2*idata[2,idx+1] + idata[3,idx+1] + idata[1,idx] - 2*idata[1,idx+1] + idata[1,idx+2] + 2 * alpha3 * (1.5 * idata[1,idx+1] - 2*idata[2,idx+1] + 0.5*idata[3,idx+1])) / 2 # Dirichlet
                        # (2 * beta * (-1 * idata[1,idx])) / 2 +
        odata[1,idx+2] = (idata[1,idx+2] - 2*idata[2,idx+2] + idata[3,idx+2] + idata[1,idx+1] - 2*idata[1,idx+2] + idata[1,idx+3] + 2 * alpha3 * (1.5 * idata[1,idx+2] - 2*idata[2,idx+2] + 0.5*idata[3,idx+2])) / 2# Dirichlet
                        # (0.5 * beta * (idata[1,idx])) / 2 + 
    end
    # sync_threads()
    if idx == Ny
        odata[1,idx] = (idata[1,idx] - 2*idata[1,idx-1] + idata[1,idx-2] + idata[1,idx] - 2*idata[2,idx] + idata[3,idx] 
                        + 2 * alpha3 * (( 1.5* idata[1,idx] - 2*idata[2,idx] + 0.5*idata[3,idx]))  ) / 4 
                        # + 2 * beta * (1.5 * idata[1,idx]) + 2 * alpha2 * (idata[1,idx]) * h
                       
        odata[1,idx-1] =  (idata[1,idx-1] - 2*idata[2,idx-1] + idata[3,idx-1] + idata[1,idx-2] - 2*idata[1,idx-1] + idata[1,idx] + 2 * alpha3 * (1.5 * idata[1,idx-1] - 2*idata[2,idx-1] + 0.5*idata[3,idx-1])) / 2 # Dirichlet
                        # (2 * beta * (-1 * idata[1,idx])) / 2 +
        odata[1,idx-2] =  (idata[1,idx-2] - 2*idata[2,idx-2] + idata[3,idx-2] + idata[1,idx-3] - 2*idata[1,idx-2] + idata[1,idx-1] + 2 * alpha3 * (1.5 * idata[1,idx-2] - 2*idata[2,idx-2] + 0.5*idata[3,idx-2])) / 2# Dirichlet
                        # (0.5 * beta * (idata[1,idx])) / 2 +
    end
    # sync_threads()
    nothing
end

function SBP_S!(idata,odata,alpha1,alpha2,alpha4,beta,h)
    (Nx,Ny) = size(idata)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    if 4 <= idx <= Ny - 3
        odata[end,idx] = (idata[3,idx] - 2*idata[2,idx] + idata[1,idx] + idata[3,idx-1] - 2*idata[3,idx] + idata[3,idx+1] + 2 * alpha4 * (1.5 * idata[3,idx] - 2*idata[2,idx] + 0.5*idata[1,idx])) / 2
    end
    # sync_threads()
    if idx == 1
        odata[end,idx] = (idata[3,idx] - 2*idata[3,idx+1] + idata[3,idx+2] + idata[3,idx] - 2*idata[2,idx] + idata[1,idx] 
                        + 2 * alpha4 * (( 1.5* idata[3,idx] - 2*idata[2,idx] + 0.5*idata[1,idx]))  ) / 4 
                        # + 2 * beta * (1.5 * idata[3,idx]) + 2 * alpha1 * (idata[3,idx]) * h
                       
        odata[end,idx+1] = (idata[3,idx+1] - 2*idata[2,idx+1] + idata[1,idx+1] + idata[3,idx] - 2*idata[3,idx+1] + idata[3,idx+2] + 2 * alpha4 * (1.5 * idata[3,idx+1] - 2*idata[2,idx+1] + 0.5*idata[1,idx+1])) / 2  # Dirichlet
                        #   (2 * beta * (-1 * idata[3,idx])) / 2  
        odata[end,idx+2] = (idata[3,idx+2] - 2*idata[2,idx+2] + idata[1,idx+2] + idata[3,idx+1] - 2*idata[3,idx+2] + idata[3,idx+3] + 2 * alpha4 * (1.5 * idata[3,idx+2] - 2*idata[2,idx+2] + 0.5*idata[1,idx+2])) / 2# Dirichlet
                        #  (0.5 * beta * (idata[3,idx])) / 2 +
    end
    # sync_threads()
    if idx == Ny
        odata[end,idx] = (idata[3,idx] - 2*idata[3,idx-1] + idata[3,idx-2] + idata[3,idx] - 2*idata[2,idx] + idata[1,idx] 
                        + 2 * alpha4 * (( 1.5* idata[3,idx] - 2*idata[2,idx] + 0.5*idata[1,idx]))  ) / 4 
                        # + 2 * beta * (1.5 * idata[3,idx]) + 2 * alpha2 * (idata[3,idx]) * h
                       
        odata[end,idx-1] = (idata[3,idx-1] - 2*idata[2,idx-1] + idata[1,idx-1] + idata[3,idx-2] - 2*idata[3,idx-1] + idata[3,idx] + 2 * alpha4 * (1.5 * idata[3,idx-1] - 2*idata[2,idx-1] + 0.5*idata[1,idx-1])) / 2 # Dirichlet
                        # (2 * beta * (-1 * idata[3,idx])) / 2 +
        odata[end,idx-2] =  (idata[3,idx-2] - 2*idata[2,idx-2] + idata[1,idx-2] + idata[3,idx-3] - 2*idata[3,idx-2] + idata[3,idx-1] + 2 * alpha4 * (1.5 * idata[3,idx-2] - 2*idata[2,idx-2] + 0.5*idata[1,idx-2])) / 2# Dirichlet
                        # (0.5 * beta * (idata[3,idx])) / 2 +
    end
    # sync_threads()
    nothing
end


function SBP_W!(idata,odata,alpha1,beta,h)
    (Nx,Ny) = size(idata)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    if 2 <= idx <= Nx-1
        odata[idx,1] = (idata[idx-1,1] - 2*idata[idx,1] + idata[idx+1,1] + idata[idx,1] - 2*idata[idx,2] + idata[idx,3] + 2 * beta * (1.5 * idata[idx,1]) + 2 * alpha1 * idata[idx,1] * h) / 2
        # (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        odata[idx,2] = (2 * beta * (-1 * idata[idx,1]))
        odata[idx,3] = (0.5 * beta * idata[idx,1])
    end
    sync_threads()
    if idx == 1 || idx == Nx
        odata[idx,1] =  (2 * beta * (1.5 * idata[idx,1]) + 2 * alpha1 * (idata[idx,1]) * h )/ 4
        odata[idx,2] = (2 * beta * (-1 * idata[idx,1])) / 2
        odata[idx,3] = (0.5 * beta * idata[idx,1]) / 2
    end
    nothing
end


function SBP_E!(idata,odata,alpha2,beta,h)
    (Nx,Ny) = size(idata)
    tidx = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tidx
    if 2 <= idx <= Nx-1
        odata[idx,3] = (idata[idx-1,3] - 2*idata[idx,3] + idata[idx+1,3] + idata[idx,3] - 2*idata[idx,2] + idata[idx,1] + 2 * beta * (1.5 * idata[idx,3]) + 2 * alpha2 * idata[idx,3] * h) / 2
        # (2 * beta * (1.5 * CPU_W_T[1,i]) + 2 * alpha2 * CPU_W_T[1,i] * h) / 2
        odata[idx,2] = (2 * beta * (-1 * idata[idx,3]))
        odata[idx,1] = (0.5 * beta * idata[idx,3])
    end
    sync_threads()
    if idx == 1 || idx == Nx
        odata[idx,3] = ( 2 * beta * (1.5 * idata[idx,3]) + 2 * alpha2 * idata[idx,3] * h ) / 4
        odata[idx,2] = (2 * beta * (-1 * idata[idx,3])) / 2
        odata[idx,1] = (0.5 * beta * idata[idx,3]) / 2
        # odata[idx,3] = 0
     end
    nothing
end

function show_nonzero_index(A)
    (M,N) = size(A)
    for i in 1:M
        for j in 1:N
            if !(isapprox(A[i,j],0,atol=1e-8))
                print((i,j))
            end 
        end
    end
    println()
end


function matrix_free_A_full_GPU(idata,odata)
    Nx,Ny = size(idata)
    h = 1/(Nx-1)
    TILE_DIM_1 = 16
    TILE_DIM_2 = 16
    griddim_2d = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim_2d = (TILE_DIM_1,TILE_DIM_2)

    alpha1 = alpha2 = -13/h
    alpha3 = alpha4 = -1
    beta = 1
    @cuda threads=blockdim_2d blocks=griddim_2d D2_split_naive_v2(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    tile_dim_1d = 256
    griddim_1d = cld(Nx,tile_dim_1d)
    GPU_OUT_W = CuArray{Float64,2}(undef,Nx,3)
    GPU_OUT_E = CuArray{Float64,2}(undef,Nx,3)
    GPU_OUT_N = CuArray{Float64,2}(undef,1,Ny)
    GPU_OUT_S = CuArray{Float64,2}(undef,1,Ny)

    @cuda threads=tile_dim_1d blocks=griddim_1d SBP_N!(view(idata,1:3,:),GPU_OUT_N,alpha1,alpha2,alpha3,beta,h)
    # println("finished N")
    @cuda threads=tile_dim_1d blocks=griddim_1d SBP_S!(view(idata,Nx-2:Nx,:),GPU_OUT_S,alpha1,alpha2,alpha4,beta,h)
    # println("finished S")
    # @cuda threads=tile_dim_1d blocks=griddim_1d SBP_W!_v2(view(idata,:,1:3),GPU_OUT_W,alpha1,beta,h)
    @cuda threads=tile_dim_1d blocks=griddim_1d SBP_W!(view(idata,:,1:3),GPU_OUT_W,alpha1,beta,h)

    # println("finished W")
    # println("Finished launching kernel for N,S,N\n")
    # @cuda threads=tile_dim_1d blocks=griddim_1d SBP_E!_v2(view(idata,:,Ny-2:Ny),GPU_OUT_E,alpha2,beta,h)
    @cuda threads=tile_dim_1d blocks=griddim_1d SBP_E!(view(idata,:,Ny-2:Ny),GPU_OUT_E,alpha2,beta,h)
    # println("finished E")

     # Copy W & E boundary
    copyto!(view(odata,1:Nx,1:3), view(odata,1:Nx,1:3) + view(GPU_OUT_W,1:Nx,1:3))
    copyto!(view(odata,1:Nx,Ny-2:Ny),view(odata,1:Nx,Ny-2:Ny) + view(GPU_OUT_E,1:Nx,1:3))

    # Copy N & S boundary
    copyto!(view(odata,1,1:Ny), view(odata,1,1:Ny) + view(GPU_OUT_N,1,:))
    copyto!(view(odata,Nx,1:Ny), view(odata,Nx,1:Ny) + view(GPU_OUT_S,1,:))

    nothing
end

function test_matrix_free_A(level;TILE_DIM_1=16,TILE_DIM_2=16)
    Nx = Ny = 2^level + 1
    memsize = sizeof(Float64) * Nx * Ny / 1024^3
    h = 1/(Nx-1)
    println("")
    println("Starting Test")
    println("2D Domain Size: $Nx by $Ny")
    @printf("memsize %f GB\n",memsize)
    Random.seed!(0)
    idata = CuArray(randn(Nx,Ny))
    odata = similar(idata)
    odata_v2 = similar(idata)
    odata_full_GPU = similar(idata)

    TILE_DIM_1 = TILE_DIM_2 = 16

   
    griddim = (cld(Nx,TILE_DIM_1),cld(Ny,TILE_DIM_2))
    blockdim = (TILE_DIM_1,TILE_DIM_2)
    @cuda threads=blockdim blocks=griddim D2_split_naive_v2(idata,odata_v2,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    @cuda threads=blockdim blocks=griddim D2_split_naive(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
    @show norm(odata-odata_v2)
    @cuda threads=blockdim blocks=griddim copy_naive!(odata,idata)
    # @cuda threads=blockdim blocks=griddim copy_naive!(d_b,d_a)
    synchronize()
    # boundary_data = boundary_containers(zeros(Nx,3),zeros(3,Nx),zeros(Nx,3),zeros(3,Nx),zeros(3,Ny),zeros(3,Ny),zeros(Nx,3),zeros(Nx,3),zeros(3,Ny),zeros(3,Ny))
    matrix_free_A(idata,odata)
    matrix_free_A_full_GPU(idata,odata_full_GPU)
    # @show odata - odata_full_GPU
    @show norm(odata-odata_full_GPU)

   



    iter_times = 1000
    @show iter_times

    # Evaluating only D2_naive
    t_D2_naive = @elapsed begin
        for _ in 1:iter_times
            @cuda threads=blockdim blocks=griddim D2_split_naive(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
        end
        synchronize()
    end
    
    t_D2_naive = t_D2_naive * 1000 / iter_times
    @show t_D2_naive
    # End evaluating D2_naive

     # Evaluating only D2_naive_v2
     t_D2_naive_v2 = @elapsed begin
        for _ in 1:iter_times
            @cuda threads=blockdim blocks=griddim D2_split_naive_v2(idata,odata,Nx,Ny,h,Val(TILE_DIM_1), Val(TILE_DIM_2))
        end
        synchronize()
    end
    
    t_D2_naive_v2 = t_D2_naive_v2 * 1000 / iter_times
    @show t_D2_naive_v2
    # End evaluating D2_naive_v2


    t_A = @elapsed begin
        for _ in 1:iter_times
            matrix_free_A(idata,odata)
        end
        synchronize()
    end
    t_A = t_A * 1000 / iter_times
    @show t_A 

    t_A_full_GPU = @elapsed begin
        for _ in 1:iter_times
            matrix_free_A_full_GPU(idata,odata)
        end
        synchronize()
    end
    t_A_full_GPU = t_A_full_GPU * 1000 / iter_times
    @show t_A_full_GPU 
    
    # through_put = 2 * memsize / (t_D2_naive_v2/1000)
    through_put = 2 * memsize / (t_A_full_GPU/1000)
    @printf("through_put:      %f GiB / s\n", through_put)

end

function CG_GPU(b_reshaped_GPU,x_GPU)
    (Nx,Ny) = size(b_reshaped_GPU);
    odata = CuArray(zeros(Nx,Ny))
    matrix_free_A(x_GPU,odata);
    r_GPU = b_reshaped_GPU - odata;
    p_GPU = copy(r_GPU);
    # rsold_GPU = sum(r_GPU .* r_GPU)
    rsold_GPU = norm(r_GPU)^2
    Ap_GPU = CuArray(zeros(Nx,Ny))
    num_iter_steps = 0
    machine_eps = sqrt(eps(real(eltype(b_reshaped_GPU))))
    rel_tol = machine_eps * max(sqrt(rsold_GPU),1)
    # @show rel_tol
    norms = [sqrt(rsold_GPU)]
    # @show rsold_GPU
    for i in 1:Nx*Ny
        num_iter_steps += 1
        matrix_free_A(p_GPU,Ap_GPU);
        # sum(p_GPU .* Ap_GPU)
        alpha_GPU = rsold_GPU / (sum(p_GPU .* Ap_GPU))
        r_GPU .-= alpha_GPU .* Ap_GPU
        x_GPU .+= alpha_GPU .* p_GPU
        rsnew_GPU = norm(r_GPU)^2
        if sqrt(rsnew_GPU) < rel_tol
            break
        end
        ratio = (rsnew_GPU/rsold_GPU)
        p_GPU .= r_GPU .+ ratio .* p_GPU;
        rsold_GPU = rsnew_GPU
        append!(norms,sqrt(rsold_GPU))
        # if i < 20
        #     @show rsold_GPU
        # end
    end
    @show 
    # @show num_iter_steps
    (num_iter_steps,norms[end])
end

function CG_GPU_v2(b_reshaped_GPU,x_GPU,odata,r_GPU,p_GPU,Ap_GPU)
    # preallocation version
    (Nx,Ny) = size(b_reshaped_GPU);
    matrix_free_A(x_GPU,odata);
    r_GPU .= b_reshaped_GPU .- odata;
    p_GPU .= r_GPU
    rsold_GPU = sum(r_GPU .* r_GPU)
    num_iter_steps = 0
    machine_eps = sqrt(eps(real(eltype(b_reshaped_GPU))))
    rel_tol = machine_eps * sqrt(rsold_GPU)
    norms = [sqrt(rsold_GPU)]
    for i in 1:Nx*Ny
        num_iter_steps += 1
        matrix_free_A(p_GPU,Ap_GPU);
        alpha_GPU = rsold_GPU / (sum(p_GPU .* Ap_GPU))
        r_GPU .-= alpha_GPU .* Ap_GPU
        x_GPU .+= alpha_GPU .* p_GPU
        # CUDA.CUBLAS.axpy!()
        rsnew_GPU = sum(r_GPU .* r_GPU)
        if sqrt(rsnew_GPU) < rel_tol
            break
        end
        p_GPU .= r_GPU .+ (rsnew_GPU/rsold_GPU) .* p_GPU;
        rsold_GPU = rsnew_GPU
        append!(norms,sqrt(rsold_GPU))
        # if i < 20
        #     @show rsold_GPU
        # end
    end
    # @show num_iter_steps
    (num_iter_steps,rel_tol,rsold_GPU)
end

function CG_full_GPU(b_reshaped_GPU,x_GPU)
    (Nx,Ny) = size(b_reshaped_GPU);
    odata = CuArray(zeros(Nx,Ny))
    # matrix_free_A_full_GPU(x_GPU,odata);
    matrix_free_A_full_GPU(x_GPU,odata)
    r_GPU = b_reshaped_GPU - odata;
    p_GPU = copy(r_GPU);
    rsold_GPU = sum(r_GPU .* r_GPU)
    # Ap_GPU = CUDA.zeros(Nx,Ny);
    Ap_GPU = CuArray(zeros(Nx,Ny))
    num_iter_steps = 0
    machine_eps = sqrt(eps(real(eltype(b_reshaped_GPU))))
    # rel_tol = machine_eps * max(sqrt(rsold_GPU),1)
    rel_tol = machine_eps
    norms = [sqrt(rsold_GPU)]
    for i in 1:Nx*Ny
    # for i in 1:20
        num_iter_steps += 1
        matrix_free_A_full_GPU(p_GPU,Ap_GPU);
        alpha_GPU = rsold_GPU / (sum(p_GPU .* Ap_GPU))
        # x_GPU = x_GPU + alpha_GPU * p_GPU;
        # r_GPU = r_GPU - alpha_GPU * Ap_GPU;
        r_GPU .-= alpha_GPU .* Ap_GPU
        x_GPU .+= alpha_GPU .* p_GPU
        # CUDA.CUBLAS.axpy!()
        rsnew_GPU = sum(r_GPU .* r_GPU)
        if sqrt(rsnew_GPU) < rel_tol
            break
        end
        p_GPU .= r_GPU .+ (rsnew_GPU/rsold_GPU) .* p_GPU;
        rsold_GPU = rsnew_GPU
        append!(norms,sqrt(rsold_GPU))
        # if i < 20
        #     @show rsold_GPU
        # end
    end
    # @show num_iter_steps
    (num_iter_steps,rel_tol,norms[end])
end

function CG_CPU(A,b,x)
    r = b - A * x;
    p = r;
    rsold = r' * r
    # Ap = spzeros(length(b))
    Ap = similar(b);

    num_iter_steps = 0
    # @show rsold
    for _ = 1:length(b)
    # for _ in 1:20
        num_iter_steps += 1
        mul!(Ap,A,p);
        alpha = rsold / (p' * Ap)
        x .= x .+ alpha * p;
        r .= r .- alpha * Ap;
        rsnew = r' * r
        if sqrt(rsnew) < sqrt(eps(real(eltype(b))))
              break
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew
        # @show rsold
    end
    # @show num_iter_steps
    num_iter_steps
end
