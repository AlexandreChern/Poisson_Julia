using CUDA

function compute(a, b)
    c = a * b             # library call
    broadcast!(sin, c, c) # Julia kernel
    c
end

function run(a, b)
    results = Vector{Any}(undef, 2)

    # computation
    @sync begin
        @async begin
            results[1] = Array(compute(a,b))
            nothing # JuliaLang/julia#40626
        end
        @async begin
            results[2] = Array(compute(a,b))
            nothing # JuliaLang/julia#40626
        end
    end

    # comparison
    results[1] == results[2]
end

function run_v2(a, b)
    results = Vector{Any}(undef, 2)

    # pre-allocate and pin destination CPU memory
    results[1] = Mem.pin(Array{eltype(a)}(undef, size(a)))
    results[2] = Mem.pin(Array{eltype(a)}(undef, size(a)))

    # computation
    @sync begin
        @async begin
            copyto!(results[1], compute(a,b))
            nothing # JuliaLang/julia#40626
        end
        @async begin
            copyto!(results[2], compute(a,b))
            nothing # JuliaLang/julia#40626
        end
    end

    # comparison
    results[1] == results[2]
end


function main(N=1024)
    a = CUDA.rand(N,N)
    b = CUDA.rand(N,N)

    # make sure this data can be used by other tasks!
    synchronize()

    run_v2(a, b)
    # run(a,b)
end

main()

