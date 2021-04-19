using JLD2

a = @load "example.jld2"

a = eval(a[1])

@show a.a
@show a.b
@show a.c