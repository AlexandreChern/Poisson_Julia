using Distributed, CUDA
addprocs(length(devices()))
@everywhere using CUDA

gpus = Int(length(devices()))

dims = (16,16)
dims_half = (dims[1],div(dims[2],2))

a = randn(dims)

b = randn(dims_half)
c = randn(dims_half)

d = randn(dims)

buf_a = Mem.alloc(Mem.Unified, sizeof(a))
d_a = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_a), dims)
finalizer(d_a) do _
    Mem.free(buf_a)
end
copyto!(d_a, a)

buf_b = Mem.alloc(Mem.Unified, sizeof(b))
d_b = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_b), dims_half)
finalizer(d_b) do _
    Mem.free(buf_b)
end
copyto!(d_b, b)

buf_c = Mem.alloc(Mem.Unified, sizeof(c))
d_c = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_c), dims_half)
finalizer(d_c) do _
    Mem.free(buf_c)
end
copyto!(d_c,c)

buf_d = Mem.alloc(Mem.Unified, sizeof(d))
d_d = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_d), dims)
finalizer(d_d) do _
    Mem.free(buf_d)
end
copyto!(d_d,d)

input_lists = [d_b, d_c]

idx = [1:8,9:16]
view(a,:,idx[1])

CUDA.allowscalar(false)



asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
        gpu = deviceid(d) + 1
        @info gpu
        # copyto!(view(d_a,:,idx[gpu]),input_lists[gpu])
        # input_lists[gpu]
        # @info d_a
        cu_A = CuArray(randn(1024,1024))
        cu_B = CuArray(randn(1024,1024))
        cu_C = similar(cu_A)
        for _ in 1:10000
            cu_C .= cu_A .* cu_B
        end
    end
end


@everywhere cu_test = CuArray(randn(3,3))

@everywhere const_a = 1.3