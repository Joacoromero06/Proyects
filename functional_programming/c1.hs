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

