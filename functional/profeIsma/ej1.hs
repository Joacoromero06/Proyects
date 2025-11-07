
main :: IO()
main = putStrLn "HOlat mundo"
miMcd:: Int -> Int -> Int
miMcd a b = if b == 0 then a else miMcd b (mod a b)

sumElems:: [Int] -> Int
sumElems []= 0
sumElems (x:xs) = x + sumElems xs

nroDigs:: Int -> Int -> Int
nroDigs b 0 = 0
nroDigs b x = 1 + nroDigs b (div x b)

fact:: Int -> Int
fact 0 = 1
fact 1 = 1
fact n = n * fact (n-1) -- (n-1) /= n-1 lazy evaluation.

