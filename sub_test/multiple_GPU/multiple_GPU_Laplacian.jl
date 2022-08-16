using CUDA
devices()

gpus = Int(length(devices()))

dims = (8,8)

a = randn(Float64,dims) .* 100


# CuArray doesn't support unified memory yet,
# so allocate our own buffers


buf_gpu1 = Mem.alloc(Mem.Unified, sizeof(a))
d_gpu1 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu1), dims)
finalizer(d_gpu1) do _
    Mem.free(buf_gpu1)
end


buf_gpu2 = Mem.alloc(Mem.Unified, sizeof(a))
d_gpu2 = unsafe_wrap(CuArray{Float64,2}, convert(CuPtr{Float64}, buf_gpu2), dims)
finalizer(d_gpu2) do _
    Mem.free(buf_gpu2)
end

copyto!(view(d_gpu1,:,1:4),view(a,:,1:4))
copyto!(view(d_gpu2,:,5:8),view(a,:,5:8))

display(d_gpu1)
display(d_gpu2)

for (gpu, dev) in enumerate(devices())
    device!(dev)
    array = Symbol("d_gpu$gpu")
    # @views d_gpu1 .= 0
    eval(array) .+= gpu
end