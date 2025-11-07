{-
A tuple (x, y, z) is called pythogerean. If x² + y² = z²
Write a function pyths that maps an integer n to all tuples pythogerean 
    such his elements are in [1..n]
-}
pyths:: Int -> [(Int, Int, Int)]
pyths n = [(x, y, z) | x <- [1..n], y <- [1..n], z <- [1..n] , (x*x) + (y*y) == (z*z)]

{-
A positive integer is perfect if it equals the sum of its factors, excludign itself
Wtire a function that returns all perfect numbers up to a limit    
-}

factors:: Int -> [Int]
factors n = [ x | x <- [1..n], mod n x == 0 ]

perfect:: Int -> Bool
perfect n = sum (tail (reverse (factors n))) == n

perfects:: Int -> [Int]
perfects n = [ x | x <- [1..n], perfect x ]

{-
Scalar product of two lists of integers of lenght n is defined as
 the sum of corresponding integers
Write a function that returns the scalar product of two lists
-}
scalarProduct:: [Int] -> [Int] -> Int
scalarProduct xs ys = sum [ x * y | (x, y) <- zip xs ys ]