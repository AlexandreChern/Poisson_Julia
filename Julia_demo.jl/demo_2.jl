n = 10
a = ones(n)
b = ones(n)
c = ones(n)
d = ones(n)

function f1(a,b,c)
    c .= a .+ b
    return c
end

function f2(a,b,c)
    c = a + b
    return c
end

function f3(a,b)
    c = a + b
    return c
end


a

b


@time f1(a,b,c)

c

@time f2(a,b,c)

function for_1(a,b,c)
    for i in 1:100
        f1(a,b,c)
    end
end

for_1(a,b,c)

function for_2(a,b,c)
    for i in 1:100
        f2(a,b,c)
    end
end

function for_3(a,b,c)
    for i in 1:100
        f3(a,b)
    end
end


@benchmark for_1(a,b,c)
@benchmark for_2(a,b,c)
@benchmark for_3(a,b,c)
