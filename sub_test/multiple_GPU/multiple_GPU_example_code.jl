using Distributed, CUDA
using Test
devices()

addprocs(length(devices()))
@everywhere using CUDA

# assign devices
asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end

gpus = Int(length(devices()))

# generate CPU data
dims = (3,4,gpus)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

# CuArray doesn't support unified memory yet,
# so allocate our own buffers
buf_a = Mem.alloc(Mem.Unified, sizeof(a))
d_a = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_a), dims)
finalizer(d_a) do _
    Mem.free(buf_a)
end
copyto!(d_a, a)

buf_b = Mem.alloc(Mem.Unified, sizeof(b))
d_b = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_b), dims)
finalizer(d_b) do _
    Mem.free(buf_b)
end
copyto!(d_b, b)

buf_c = Mem.alloc(Mem.Unified, sizeof(a))
d_c = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_c), dims)
finalizer(d_c) do _
    Mem.free(buf_c)
end

for (gpu, dev) in enumerate(devices())
    device!(dev)
    @views d_c[:, :, gpu] .= d_a[:, :, gpu] .+ d_b[:, :, gpu]
end

for dev in devices()

    # NOTE: normally you'd use events and wait for them
    device!(dev)
    synchronize()
end


c = Array(d_c)
@test a+b ≈ c