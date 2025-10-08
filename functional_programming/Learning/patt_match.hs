
{-
definir funciones según la estructura de sus argumentos.
f patron1 = exp1
f patron2 = exp2
hs prueba los patrones en orden
patrones no ifs. -> forma del dato. -> no condiciones
-}
fact 0 = 1
fact n = n*(fact n-1)

{-
Un guard en haskell es una condicion booleana
Se coloca a lado de la definicion de una funcion
Esta precedida por |
Sintaxis
f args 
| cond1 = <expr>
| cond2 = <expr>
| otherwise = <>expr
Cada condicion se evalua en orden y solo si es necesario
otherwise es un true definido en el Prelude
-}

--funcion: primer digito de una lista es pos, neg o cero
signOfHead:: [Integer] -> String
signOfHead [] = "lista vacia"
signOfHead (x:xs) 
    | x > 0 = "positivo"
    | x < 0 = "negativo"
    | otherwise = "nulo"