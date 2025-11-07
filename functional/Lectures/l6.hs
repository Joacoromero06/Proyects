{-
Write a function stail, that behaves as tail, except that maps [] -> []
-}
stail1:: [Int] -> [Int]
stail1 xs = if null xs then mat_2_list [] else tail xs

stail2:: [Int] -> [Int]
stail2 [] = mat_2_list []
stail2 (_:xs) = xs

stail3:: [Int] -> [Int]
stail3 xs
 | null xs = mat_2_list []
 | otherwise = tail xs

{-
Give 3 definition of ' || ' or operator using pattern matching
-}
myOr1:: Bool -> Bool -> Bool
myOr1 False True  = True
myOr1 False False = False
myOr1 True  True  = True
myOr1 True  False = True

myOr2:: Bool -> Bool -> Bool
myOr2 False False = False
myOr2 _     _     = True

myOr3:: Bool -> Bool -> Bool
myOr3 False b = b
myOr3 True  _ = True

{-
Redifine the function and '&&' using conditionals
True && True = True
_    && _    = False
-}
myAnd:: Bool -> Bool -> Bool
myAnd b1 b2 = if b1 && b2 then True else False