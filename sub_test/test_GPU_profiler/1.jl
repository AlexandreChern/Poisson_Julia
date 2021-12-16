using CUDA

N = 1024

A = CuArray(randn(N,N))

B = CuArray(randn(N,N))


C = A + B
D = A * B


A \ B