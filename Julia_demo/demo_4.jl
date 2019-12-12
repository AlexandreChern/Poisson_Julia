using Parameters

@with_kw struct vars
    a = 1
    b = 2
    c = 3
end

var = vars()


function f1(a,b,c)
    return c = a+b
end

function f2(a,b,c)
    return c = a-b
end

function g(var)
    @unpack a,b,c = var
    u = f1(a,b,c) * f2(a,b,c)
    return u
end

g(var)

a # not defined
