{-
una listaen hs es una una coleccion de elementos del mismo tipo 
ordenada.Applicative
Si el tipo de los elementos es t el de la lista es [t]
DIstingue de orden y los repetidos
-}

--COncatenacion '++' es un operador
--probar con listas y String de ""
-- [a1,..,an]!!posicion, da error si no esta la posicion

{-
Construccion de listas
EStanda: [a1, ..., an]
cons: usa ':
-}

{-

func [int] -> type
func (x:xs)
-}

{-
[ expresión | generador, condición ]
[ (x,y) | x <- xs, y<-ys ,x>0, even x ]
-}

-- clase
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}
swap:: (Int, Float) -> (Float, Int)
swap (x_int, x_float) = (x_float, x_int)

tup2list:: (Int,Int) -> [Int]
tup2list (a, b) = [a..b]

-- palindromo:: String -> Bool
{-palindromo s 
    (elem_s:xs) = elem_s == tail xs && palindromo xs--sacarle ultimo
    "" = True-}

insert_tup:: (char, char)-> String -> String
insert_tup (a, b) s = [a] ++ s ++ [b] 