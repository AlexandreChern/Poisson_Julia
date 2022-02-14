# using CuArrays
#using CUDAnative
#using CUDAdrv: synchronize,  device
using CUDA
using Printf
using StaticArrays
using KernelAbstractions.Extras: @unroll

function transpose_cpu!(b, a)
    N = size(a, 1)
    for j = 1:N
        for i = 1:N
            b[j, i] = a[i, j]
        end
    end
end

function transpose_naive!(b, a)
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
        # @inbounds b[j, i] = a[i, j]
        @inbounds b[i, j] = a[j, i]
    end

    return nothing
end

function transpose_strided!(b, a, ::Val{TILE_DIM}, ::Val{STRIDE}) where {TILE_DIM, STRIDE}
    N = size(a, 1)

    tidx, tidy = threadIdx().x, threadIdx().y

    # which block of threads are we in
    bidx = blockIdx().x
    bidy = blockIdx().y

    # what index am I in the global thread space
    i0 = tidx + TILE_DIM * (bidx - 1)
    j  = tidy + TILE_DIM * (bidy - 1)

    T = eltype(a)
    # NUM_ELEM will be known at compile, since TILE_DIM and STRIDE are known at compile time
    NUM_ELEM = div(TILE_DIM, STRIDE)
    # Static sized array -> stored in registers
    tile = MArray{Tuple{NUM_ELEM}, T}(undef)

    # If we are inside the data
    if j <= N
        # Loop over the number of elements we process and load the data
        @unroll for k = 0:NUM_ELEM-1
            # Shift each time by STRIDE
            i = i0 + k * STRIDE
            if i <= N
                @inbounds tile[k+1] = a[j, i]
            end
        end
        # Loop over the number of elements we process and write the data
        @unroll for k = 0:NUM_ELEM-1
            # Shift each time by STRIDE
            i = i0 + k * STRIDE
            if i <= N
                @inbounds b[i, j] = tile[k+1]
            end
        end
    end
    nothing
end

function transpose_shared!(b, a, ::Val{TILE_DIM}) where TILE_DIM
    N = size(a, 1)

    # which thread are we in our block
    tidx = threadIdx().x
    tidy = threadIdx().y

    # which block of threads are we in
    bidx = blockIdx().x
    bidy = blockIdx().y

    # what index am I in the global thread space
    i = tidx + TILE_DIM * (bidx - 1)
    j = tidy + TILE_DIM * (bidy - 1)

    # TILE_DIM needs to be known at compile time, hence we use a Val
    tile = @cuStaticSharedMem(eltype(a), (TILE_DIM, TILE_DIM))

    if i <= N && j <= N
        # Tile is a matrix of size dimx X dimy
        @inbounds tile[tidx, tidy] = a[i, j]
    end

    i = tidx + TILE_DIM * (bidy - 1)
    j = tidy + TILE_DIM * (bidx - 1)

    # All threads in the block do not pass this point until all the threads are
    # here
    sync_threads()

    if i <= N && j <= N
        @inbounds b[i, j] =  tile[tidy, tidx]
    end

    return nothing
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

function copy_naive!_v2(odata,idata,Nx,Ny,::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    #Nx = size(idata,1)
    #Ny = size(idata,2)
    tidx = threadIdx().x
    tidy = threadIdx().y

    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    if i <= Nx && j <= Ny
           @inbounds odata[i,j] = idata[i,j]
    end

    nothing
end


# Arguments after semicolon are keyword arguments
function main(; N = 1024, FT = Float32, tile_dim = 32, stride = 8, num_reps = 1000)
    memsize = N * N * sizeof(FT) / 1024^3
    println(device())
    @printf("Float type:      %s\n", FT)
    @printf("Matrix size:     %4d x %4d\n", N, N)
    @printf("Memory required: %f GiB\n", memsize)

    # Host arrays
    Random.seed!(0)
    a = rand(FT, N, N)
    b = similar(a)

    # transpose_cpu!(b, a)
    # @assert b == a'

    # device arrays
    # Creating the device array and then copying a to the device array
    # d_a = CuArray{FT, 2}(undef, N, N)
    # copy!(d_a, a)
    d_a = CuArray(a)
    d_b = similar(d_a)

    # Copy Naive
    nblocks = (cld(N, tile_dim), cld(N, tile_dim))
    @cuda threads=(tile_dim, tile_dim) blocks=nblocks copy_naive!(d_b, d_a)
    @cuda threads=(tile_dim, tile_dim) blocks=nblocks copy_naive!_v2(d_b,d_a,N,N,Val(tile_dim),Val(tile_dim))
    @assert Array(d_b) == a
    synchronize()
    time = @elapsed begin
        for _ = 1:num_reps
            @cuda threads=(tile_dim, tile_dim) blocks=nblocks copy_naive!(d_b, d_a)
            # @cuda threads=(tile_dim,tile_dim) blocks=nblocks copy_naive!_v2(d_b,d_a,N,N,Val(tile_dim),Val(tile_dim))
        end
        synchronize()
    end
    avg_time = time / num_reps
    println("average time: $avg_time")
    bndw = 2 * memsize / avg_time
    @printf("copy_naive:      %f GiB / s\n", bndw)

    # Transpose Naive
    nblocks = (cld(N, tile_dim), cld(N, tile_dim))
    @cuda threads=(tile_dim, tile_dim) blocks=nblocks transpose_naive!(d_b, d_a)
    @assert Array(d_b) == a'
    synchronize()
    time = @elapsed begin
        for _ = 1:num_reps
            @cuda threads=(tile_dim, tile_dim) blocks=nblocks transpose_naive!(d_b, d_a)
        end
        synchronize()
    end
    avg_time = time / num_reps
    bndw = 2 * memsize / avg_time
    @printf("transpose_naive: %f GiB / s\n", bndw)


    # Transpose Shared
    nblocks = (cld(N, tile_dim), cld(N, tile_dim))
    fill!(d_b, 0)
    @cuda threads=(tile_dim, tile_dim) blocks=nblocks transpose_shared!(d_b, d_a, Val(tile_dim))
    @assert Array(d_b) == a'
    synchronize()
    time = @elapsed begin
        for n = 1:num_reps
            @cuda threads=(tile_dim, tile_dim) blocks=nblocks transpose_shared!(d_b, d_a, Val(tile_dim))
        end
        synchronize()
    end
    avg_time = time / num_reps
    bndw = 2 * memsize / avg_time
    @printf("transpose_shared: %f GiB / s\n", bndw)


    # Transpose Strided
    nblocks = (cld(N, tile_dim), cld(N, tile_dim))
    fill!(d_b, 0)
    @cuda threads=(stride, tile_dim) blocks=nblocks transpose_strided!(d_b, d_a, Val(tile_dim), Val(stride))
    @assert Array(d_b) == a'
    synchronize()
    time = @elapsed begin
        for n = 1:num_reps
            @cuda threads=(stride, tile_dim) blocks=nblocks transpose_strided!(d_b, d_a, Val(tile_dim), Val(stride))
        end
        synchronize()
    end
    avg_time = time / num_reps
    bndw = 2 * memsize / avg_time
    @printf("transpose_strided!: %f GiB / s\n", bndw)

end
