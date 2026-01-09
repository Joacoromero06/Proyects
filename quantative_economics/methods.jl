function f(x::Number)
    println("Number function invoked")
end
function f(x::Int64)
    println("Int64 function invoked")
end
function f(x)
    println("Generic function invoked")
end
