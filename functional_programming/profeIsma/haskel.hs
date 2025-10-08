{-
1. Todo es valor

En Haskell:

3 es un valor del tipo Int.
"hola" es un valor del tipo String.
doble es un valor del tipo Int -> Int.
Incluso \x -> x+1 es un valor, del tipo Num a => a -> a.

👉 No hay una distinción ontológica fuerte entre “datos” y “funciones”: ambos son valores, solo que de diferentes tipos. Eso es lo que permite que una función se pase como argumento o se devuelva como resultado
-}

{-
Haskell no tiene “sentencias” (como un for o if imperativo que no devuelve nada), 
    sino expresiones que siempre producen un valor:
        3+4 → expresión aritmética que evalúa a un valor 7.
        if True then 1 else 2 → expresión condicional que evalúa a 1.
        doble 5 → expresión funcional que evalúa a 10.
Una función sin aplicar (doble) también es una expresión, que se evalúa a la propia función como valor.
-}

{-
Una función define una relación entre entradas y salidas (por medio de una ecuación).

A la vez, define una transformación: un mecanismo que toma un valor y devuelve otro.

Haskell no obliga a elegir una sola interpretación; su semántica formal
 (basada en cálculo lambda y reescritura de expresiones) unifica estas visiones.
-}

{-
Todo programa es una expresión.

Toda expresión tiene un tipo.

El tipo dice qué clase de valor va a resultar cuando esa expresión se evalúe.
-}