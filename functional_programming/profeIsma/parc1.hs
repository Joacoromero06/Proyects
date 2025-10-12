
import Distribution.Compat.Lens (_2)
import Distribution.TestSuite (Result(Error))
rellenar:: String -> Int -> String
rellenar xs n
 | mod (length xs) n == 0 = xs
 | otherwise = xs ++ replicate (n - mod (length xs) n) ' '

list_2_mat:: String -> Int -> [String]
list_2_mat [] _ = []
list_2_mat xs n = take n xs : list_2_mat (drop n xs) n 

mat_2_list:: [String] -> String
mat_2_list []       = []
mat_2_list (xs:xss) = xs ++ mat_2_list xss

transp_mat:: [String] -> [String]
transp_mat xss = [ [head xs | xs <- xss] ] ++ transp_mat [ tail xs | xs <- xss ]

quicksort:: [Int] -> [Int]
quicksort []     = []
quicksort (piv:xs) = menores ++ [piv] ++ mayores
 where menores = [x | x <- xs, x < piv]
       mayores = [x | x <- xs, x > piv]


my_min:: Int -> Int -> Int
my_min a b 
 | a <= b    = a
 | otherwise = b

my_max:: Int -> Int -> Int
my_max a b 
 | a >= b    = a
 | otherwise = b

merge:: [Int] -> [Int] -> [Int]
merge xs      []     = xs
merge []     ys      = ys
merge (x:xs) (y:ys) 
 | x <= y    = x : merge xs (y:ys)
 | otherwise = y : merge (x:xs) ys 

{-
Una lista de un solo elemento ya esta ordenada
Divide una lista en 2 las ordena y 
-}
mergesort:: [Int] -> [Int]
mergesort []      = []
mergesort xs
 | length xs == 1 = xs
 | otherwise      = merge (take mitad_1 xs) (drop mitad_2 xs) 
 where 
    mitad_1 =
 

biseccion:: (Float -> Float) -> (Float, Float) -> Float -> Float
biseccion f (a,b) eps
 | f a * f b > 0    = error "hola mal el intervalo"
 | abs (f m) < eps  = m
 | f a * f m < 0    = biseccion f (a, m) eps
 | otherwise        = biseccion f (m, b) eps
  where 
    m = (a + b) / 2


{-
Un polinomio es un 
-}