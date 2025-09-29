(define (gcd u v)
    (if (= v 0) u
        (gcd v (modulo u,v) )
    )
)

#(fn arg1 arg2) 
# Todos son listas en lisp.
# Al evaluar una lista en lisp:
    el primer elemento debe ser una funcion y los siguientes los argumentos
# Todos son funciones -> retornan algo

#(if a b c) si a then b else c
# REpresenta el control como el valor del computo
# Lazy evaluation primero se evalua a, luego b o c