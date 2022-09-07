using CUDA
using Test


function vadd(gpu,a,b,c)
    i = threadIdx().x + blockDim().x * ((blockIdx().x-1) + (gpu-1) * gridDim().x)
    c[i] = a[i] + b[i]
    return
end


gpus = Int(length(devices()))

dims = (gpus,32,32)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)


# FIXME: CuArray doesn't tie in with unified memory yet
# buf_a = Mem.alloc(sizeof(a), true)
# Mem.upload!(buf_a, a)
# d_a = CuArray{Float32,3}(buf_a, dims)
# buf_b = Mem.alloc(sizeof(a), true)
# Mem.upload!(buf_b, b)
# d_b = CuArray{Float32,3}(buf_b, dims)
# buf_c = Mem.alloc(sizeof(a), true)
# d_c = CuArray{Float32,3}(buf_c, dims)


buf_a = Mem.alloc(Mem.Unified, sizeof(a))
d_a = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_a), size(a))
finalizer(d_a) do _
    Mem.free(buf_a)
end
copyto!(d_a, a)

buf_b = Mem.alloc(Mem.Unified, sizeof(b))
d_b = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_b), size(b))
finalizer(d_b) do _
    Mem.free(buf_b)
end
copyto!(d_b, b)

buf_c = Mem.alloc(Mem.Unified, sizeof(a))
d_c = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, buf_c), size(a))
finalizer(d_c) do _
    Mem.free(buf_c)
end
copyto!(d_c,a)


len = prod(dims)
blocks = gpus
threads = len ÷ blocks

for (gpu,dev) in enumerate(devices())
    @debug "Allocating slice $gpu on device $(name(dev))"
    device!(dev)
    for _ in 1:1000
        @cuda blocks=blocks÷gpus threads=threads vadd(gpu, d_a, d_b, d_c)
    end
end


@debug "Synchronizing devices"
for dev in devices()
    # NOTE: normally you'd use events and wait for them
    device!(dev)
    synchronize()
end

gpu_devices = Dict(enumerate(devices()))



# Now using distributed process
# @everywhere gpus = Int(length(devices()))
# @everywhere dims = (gpus,32,32)
# @everywhere len = prod(dims)
# @everywhere blocks = gpus
# @everywhere threads = len ÷ blocks
# @everywhere function vadd(gpu,a,b,c)
#     i = threadIdx().x + blockDim().x * ((blockIdx().x-1) + (gpu-1) * gridDim().x)
#     c[i] = a[i] + b[i]
#     return
# end

# now using Threads.@spawn This is still not showing asynchronous execution

@sync begin
    Threads.@spawn begin
        # device!(0)
        gpu = 1
        device!(gpu_devices[1])
        for _ in 1:1000
            @cuda blocks=blocks÷gpus threads=threads vadd(gpu, d_a, d_b, d_c)
        end
    end
    Threads.@spawn begin
        gpu = 2
        device!(gpu_devices[2])
        for _ in 1:1000
            @cuda blocks=blocks÷gpus threads=threads vadd(gpu, d_a, d_b, d_c)
        end
    end
end

c = Array(d_c)
@test a+b ≈ c


# @sync begin
#     Threads.@spawn begin
#         device!(0)
#         for _ in 1:10000
#             @cuda blocks=blocks÷gpus threads=threads vadd(gpu, d_a, d_b, d_c)
#     end
#     Threads.@spawn begin
#         device!(1)
#         for _ in 1:10000
#             @cuda blocks=blocks÷gpus threads=threads vadd(gpu, d_a, d_b, d_c)
#         end
#     end
# end