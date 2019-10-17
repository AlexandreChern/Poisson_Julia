n = 10

a = zeros(n)

b = a

pointer(a)

pointer(b)

a .+= 1

a

b

pointer(a)

pointer(b)

a .+= ones(n)

pointer(a)

pointer(b)

a += ones(n)

pointer(a)

pointer(b)

a

b

a .+= 1

a

b


using BenchmarkTools

@benchmark $a .+= 1

@benchmark $a .+= ones(n)

@benchmark $a += ones(n)
