# This is old code
using CUDArt, CUSPARSE  ## note: modified version of CUSPARSE, as indicated above.

N = 10^3;
M = 10^6;
p = 0.1;

devlist = devices(dev->true);
nGPU = length(devlist)

dev_X = Array(CudaSparseMatrixCSR, nGPU)
dev_b = Array(CudaArray, nGPU)
dev_c = Array(CudaArray, nGPU)
Handles = Array(Array{Ptr{Void},1}, nGPU)


for (idx, dev) in enumerate(devlist)
    println("sending data to device $dev")
    device(dev) ## switch to given device
    dev_X[idx] = CudaSparseMatrixCSR(sprand(N,M,p))
    dev_b[idx] = CudaArray(rand(M))
    dev_c[idx] = CudaArray(zeros(N))
    Handles[idx] = cusparseHandle_t[0]
    cusparseCreate(Handles[idx])
end


function Pmv!(
    Handle::Array{Ptr{Void},1},
    transa::SparseChar,
    alpha::Float64,
    A::CudaSparseMatrixCSR{Float64},
    X::CudaVector{Float64},
    beta::Float64,
    Y::CudaVector{Float64},
    index::SparseChar)
    Mat     = A
    cutransa = cusparseop(transa)
    m,n = Mat.dims
    cudesc = getDescr(A,index)
    device(device(A))  ## necessary to switch to the device associated with the handle and data for the ccall 
    ccall(
        ((:cusparseDcsrmv),libcusparse), 

        cusparseStatus_t,

        (cusparseHandle_t, cusparseOperation_t, Cint,
        Cint, Cint, Ptr{Float64}, Ptr{cusparseMatDescr_t},
        Ptr{Float64}, Ptr{Cint}, Ptr{Cint}, Ptr{Float64},
        Ptr{Float64}, Ptr{Float64}), 

        Handle[1],
        cutransa, m, n, Mat.nnz, [alpha], &cudesc, Mat.nzVal,
        Mat.rowPtr, Mat.colVal, X, [beta], Y
    )
end

function test(Handles, dev_X, dev_b, dev_c, idx)
    Pmv!(Handles[idx], 'N',  1.0, dev_X[idx], dev_b[idx], 0.0, dev_c[idx], 'O')
    device(idx-1)
    return to_host(dev_c[idx])
end


function test2(Handles, dev_X, dev_b, dev_c)

    @sync begin
        for (idx, dev) in enumerate(devlist)
            @async begin
                Pmv!(Handles[idx], 'N',  1.0, dev_X[idx], dev_b[idx], 0.0, dev_c[idx], 'O')
            end
        end
    end
    Results = Array(Array{Float64}, nGPU)
    for (idx, dev) in enumerate(devlist)
        device(dev)
        Results[idx] = to_host(dev_c[idx]) ## to_host doesn't require setting correct device first.  But, it is  quicker if you do this.
    end

    return Results
end

## Function times given after initial run for compilation
@time a = test(Handles, dev_X, dev_b, dev_c, 1); ## 0.010849 seconds (12 allocations: 8.297 KB)
@time b = test2(Handles, dev_X, dev_b, dev_c);  