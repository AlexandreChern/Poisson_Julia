using JLD2

struct test_struct
    a::Int64
    b::Array{Float64,1}
    c::Array{Float64,2}
end

b = test_struct(3,randn(3),randn(3,3))

@save "example.jld2"

# @load "example.jld2" 