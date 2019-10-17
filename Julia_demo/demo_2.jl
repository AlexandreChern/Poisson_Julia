n = 100
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

function f4(a,b)
    c .= a .+ b
    return c
end

a

b

@benchmark f1(a,b,c)

@benchmark f2(a,b,c)

@benchmark f3(a,b)


function for_1(a,b,c)
    for i in 1:1000
        f1(a,b,c)
    end
end

for_1(a,b,c)

function for_2(a,b,c)
    for i in 1:1000
        f2(a,b,c)
    end
end

function for_3(a,b)
    for i in 1:1000
        f3(a,b)
    end
end


@benchmark for_1(a,b,c)
@benchmark for_2(a,b,c)
@benchmark for_3(a,b,c)
