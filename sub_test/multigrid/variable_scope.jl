function foo(a)
        b = 3
        global c = a + b
        @show c
        bar() 
end

function bar()
   for i in 1:c
        println("$i")
   end
end
