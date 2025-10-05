--name_fn arg1 arg2 ....argn = <expr>
-- for aplication: name_fn arg1 ... argn
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}

in_range l_l x l_u = 
    x <= l_u && x >= l_l

{-
every VALUE has a TYPE
name :: <type>
-}
x::Int
x = 1

band:: Bool
band = True

y = 3.1415

-- FUNCTIONS ARE JUST VALUES
in_range2:: Integer -> Integer -> Integer -> Bool
in_range2 l_l x l_u = 
    x <= l_u && x >= l_l

{-
HOW to do:
    in_range = lambda min x max: min <= x <= max PYTHON
    in_range(min, x, max) -> return min <= x <= max OTHER PL

LET & BINDING EXPRESSIONS
LET letus to BIND a RESULT of an EXPRESSION to a NAME
-}
in_range3:: Integer -> Integer -> Integer -> Bool
in_range3 min x max =  
    let low_bound = min <= x
        up_bound = x <= max
    in
        low_bound && up_bound

{-
WHERE & BINDING EXPRESSIONS
WHERE letus define the FUNCTION EXPRESSION with NAMES WHERE
the NAMES are defined with certain EXPRESSION
-}
in_range4:: Integer -> Integer -> Integer -> Bool
in_range4 min x max = l_low && l_up
    where
        l_low = min <= x
        l_up = x <= max

es_pos:: Int -> Bool
es_pos x = x>0

-- % a b = a `mod` b

es_devisible:: Int -> Int -> Bool
es_devisible n x = mod x n == 0

-- suma los numeros pares de una lista de enteros
sumaPares:: [Int] -> Int
sumaPares (x:xs)   
 |mod x 2 == 0 = x + sumaPares xs
 |mod x 2 == 1 = sumaPares xs
sumaPares [] = 0


-- transforma un caracter a una lista de n caracteres
replicar:: Char -> Int -> String
replicar c 0 = []
replicar c n = c: replicar c (n-1)

-- Reduce un String a su cantidad de caracteres
cuentaVocales:: String -> Int
cuentaVocales [] = 0
cuentaVocales (x:xs)
 | x == 'a' = 1 + cuentaVocales xs
 | x == 'e' = 1 + cuentaVocales xs
 | x == 'i' = 1 + cuentaVocales xs
 | x == 'o' = 1 + cuentaVocales xs
 | x == 'u' = 1 + cuentaVocales xs
 | otherwise = cuentaVocales xs
-- definir primero los casos bases y errores

-- Transforma un String a su forma donde no comparte caracteres con otro String
limpia:: String -> String -> String
limpia x y

