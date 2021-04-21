using JLD2

a = @load "example.jld2"

b = eval(a[1])

@show b.a
@show b.b
@show b.c

c = eval(a[2])
@show c.a
@show c.b
@show c.c
