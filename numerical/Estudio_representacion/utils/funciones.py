"""
Módulo de funciones auxiliares para conversiones numéricas
"""
from collections import deque
import os

def parte_entera(x: float) -> int:
        return  int(x)

def parte_decimal(x: float) :
    """
    Descripcion:
        Dado un flotante x de python, computamos en un entero de
        python su parte decimal
    Input:
        x: numero float a obtener su parte decimal
    Output:
        x_dec: un entero que ES la parte decimal de x
    """
    x = x - parte_entera(x)
    x_dec = 0
    band = True
    cant_ceros_al_principio = 0

    while x != 0: 
        x *= 10
        dig_dec = parte_entera(x)
        if band:
            if dig_dec == 0:
                cant_ceros_al_principio += 1
            else:
                band = False

        x_dec = x_dec * 10 + dig_dec
        x = x - dig_dec

    return (x_dec, cant_ceros_al_principio)

def cant_dig(x: int) -> int:
    """
    Descripcion:
        COmputa la cantidad de digitos de un entero de python
    Input:
        x: un entero de python donde, x = d1d2...dn
    Output:
        n: la cantidad de digitos de x
    """
    n = 0
    while x != 0 :
        x = x // 10
        n += 1
    return n

def int2list(x: int) -> list:
    """
    Descripcion:
        Dado un entero x creamos una lista de python
        con todos sus digitos.
    Input:
        x: un entero del estilo x = d1d2...dn
    Output:
        x_lista: una lista pythonic donde
        x_lista = [d1, d2 ,...,dn ]
    """
    x_lista: deque = deque()
    d_i: int
    while x != 0:
        d_i = x % 10
        x_lista.appendleft( d_i )
        x = x // 10
    return list(x_lista)

def list2str(x_lista: list) -> str:
    """
    Descripcion:
        Dado una lista de python x, convertimos sus elementos
        en toda un string concatenado sin espacion
    Input:
        x: un list de la forma x = [elem_1, elem_2, ...elem_n]
    Output:
        s: un str de la forma s = 'elem_1elem_2...elem_n'
    """
    return ''.join(str(elem) for elem in x_lista)

def list_dig2int(x_lista: list) -> int:
    """
    Descripcion:
        Dado una lista quie posee solo digitos, computamos el
        entero que representaria esa lista
    Input:
        x_lista: una list de la forma x = [d1,d2,..., dn] con di un digito
    Output:
        x_ent: un int de la forma d1d2...dn
    """
    x_ent = 0
    for d_i in x_lista:
        x_ent = x_ent * 10 + d_i 
    return x_ent

def division_reiterada(x_ent: int, B:int) -> list:
    """
    Descripcion:
        Dado un digito entero 10 computa x_B una
        lista de digitos en base B donde son equivalentes
        usando division reiterada
    Input:
        x_ent: 
            type = int
            x_ent = di en base 10
    Output
        x_B:
            type = list()
            x_lista = [d1,d2,...,dn], di en base B
    """
    x_B = deque()
    while x_ent >= B:
        x_B.appendleft(x_ent % B)
        x_ent = x_ent // B
    x_B.appendleft(x_ent)
    return list(x_B)

def list10_listB(x_lista: list, B: int) -> list:
    """
    Descripcion:
        Dado un list de digitos en base 10 computa x_B una
        lista de digitos en base B donde son equivalentes
        usando division reiterada
    Input:
        x_lista: 
            type = list()
            x_lista = [d1,d2,...,dn], di en base 10
    Output
        x_B:
            type = list()
            x_lista = [d1,d2,...,dn], di en base B
    """
    x_ent = list_dig2int(x_lista)    
    return division_reiterada(x_ent, B)

def primer_digito(x_ent: int ) -> int:
    x = abs(x_ent)
    while x >= 10:
        x = x // 10
    return x

def n_primeris_digitos(x_ent: int, n: int) -> int:
    digitos = 0
    for i in range (n):
        digitos = digitos * 10 + primer_digito(x_ent) 
        x_ent = quitar_1er_digito(x_ent)
    return digitos  

def quitar_1er_digito(x_ent: int) -> int:
    x = 0
    e = 0
    while x_ent >= 10:
        x = x + (x_ent % 10) * (10**e)
        x_ent = x_ent // 10
        e = e + 1
    return x   

def producto(x_lista: list, dig: int) -> list:
    prod = list()
    acarreo = 0
    for i in range(len(x_lista)):
        xi_x_dig = x_lista[i] * dig
        en_i = xi_x_dig + acarreo
        acarreo = primer_digito(en_i) if en_i >= 10 else 0
        prod.append( en_i %10 )
    
    if acarreo != 0:
        prod.append(acarreo)

    return prod    

def suma(x: list, y) -> list:
    suma = list()
    dif = len(x) - len(y)
    n = max( len(x), len(y) )
    if dif != 0:
        y.extend([0]*dif) if dif > 0 else x.extend([0]*abs(dif))

    acarreo = 0
    for i in range(n):
        xi_mas_yi = x[i] + y[i]
        en_i = xi_mas_yi + acarreo
        suma.append( en_i % 10 )
        acarreo = 1 if en_i >= 10 else 0
    
    if acarreo != 0:
        suma.append( acarreo )

    return suma

def producto_B(x_lista: list, B: int) -> list:
    """
    Descripcion:
        Computa el producto que representa realizar el numero representado por x_lista
        un numero decimal, con B un entero
    Input:
        x_lista:
            type = list o deque
            x_lista = [dn,dn-1,....,d1,COMA,0]
        B:
            type = int
            B = 2, 3, 16, 8 bases comunmente usadas
    Output:
        lis_producto:
            type = list o deque
            lis_producto = [cn,cn-1,...,c1,COMA,s1,..,si]
    """
    lis_producto =list()
    cd = cant_dig(B)
    for i in range(cd):
        #lis_producto.extend( [0]*i ) lo hace suma
        prod_1_dig = deque(producto(x_lista, B % 10))
        prod_1_dig.extendleft([0]*i)
        lis_producto = suma( lis_producto, prod_1_dig )
        B = B // 10
    return lis_producto

def multiplicacion_reiterada(x_lista: list, B: int, t_restante: int) -> list:
    """
    Descripcion:
        Dado un x_lista una lista que representa la parte decimal de un numero, 
        computamos la multiplicacion reiterada, formando el numero equivalente en
        la base B.
    Input:
        x_lista:
            type = list
            Permite manejar el numero como en papel
        B:
            type = int
            La base a donde calculamos
        t_restante:
            type = int
            El limite para evitar multiplicar infinitas veces
            Se relaciona con la maya y la parte entera
    """
    x_B = list()
    x = x_lista[::-1]
    # pos_coma = 3. Con x =[d4,d3,d2,d1,d0] -> d0d1,d2d3d4
    pos_coma = len(x) - 1
    tier_breaking = False
    
    for i in range (t_restante):
        if list_dig2int((x[:pos_coma])[::-1]) == 0:
            tier_breaking = True
            break
        aux = producto_B(x, B)
        x, dig_x_B = aux[:pos_coma:], aux[pos_coma:]

        #Debugging
        print(f'\n\nDebugging:\nIteracion: {i}-sima\nProducto x Base: {aux}\nParte decimal: {x}\nParte entera: {dig_x_B}')
        x_B.append( list_dig2int(dig_x_B[::-1]) )
    

    while not tier_breaking:
        aux = producto_B(x, B)
        x, dig_x_B = aux[:pos_coma:], aux[pos_coma:]
        if B % 2 == 0:
            if list_dig2int(dig_x_B[::-1]) >= B // 2:
                tier_breaking = True
                redondeo_up = True
            else:
                tier_breaking = True
                redondeo_up = False
        else:
            if list_dig2int(dig_x_B[::-1]) >= B // 2 + 1:
                tier_breaking = True
                redondeo_up = True
            elif list_dig2int(dig_x_B[::-1]) < B // 2 :
                tier_breaking = True
                redondeo_up = False

    if redondeo_up:
        # Esta suma es de punto flotante.
        print('redondeo')
        unit_last_place = [0] * (len(x_B) - 1) 
        unit_last_place.extend([1])
        print(unit_last_place)
        x_B = suma( x_B, unit_last_place  )
        
    return x_B

   

__all__ = ['parte_entera', 'parte_decimal', 'cant_dig', 
           'int2list', 'list2str', 'list_dig2int',
           'division_reiterada', 'list10_listB', 'primer_digito']

if __name__ == '__main__':
    x = [0,2]
    B = 3
    
    print(f' Multplicacion reirada\n {multiplicacion_reiterada(x, B, 6)}')
"""
x = [3,2,0,1,0,0,0]
    
    print(f'producto x con 1: {producto(x,1)}\n se lo corre con -')
    print(f'producto x con 6: {producto(x,6)}')
    aux = deque(producto(x,1))
    aux.extendleft([0]); aux = list(aux)
    print(f'producto x con 1: {aux}')
    print(f'\nsumandolos: {suma(list(aux),producto(x,6))} ')
    
    aux = deque(producto(prod_1,1))
    aux.extendleft([0])
    print('Paso a paso')
    print(producto(prod_1,6))
    print(producto(list(aux),1))

    
    """

    
"""
    band = False
    tier_breaking = True

    for i in range(t_restante):
        x_anterior = x_ent
        x_ent = x_ent * B
        x_B.append(primer_digito(x_ent))
        x_ent = quitar_1er_digito(x_ent)
        if x_ent == 0:
            x_B.extend( [0] * (t_restante - i + 1) )
            band = True
            break
        
    if not band:
        while not tier_breaking:
            x_ent = x_ent * B
            if primer_digito( x_ent ) > B//2:
                tier_breaking = True
                redondeo = 1
            if primer_digito( x_ent ) < B//2:
                tier_breaking = True
                redondeo = 0
            x_ent = quitar_1er_digito(x_ent)
    return x_B
"""
"""
x_B = list()
    n = len(x_lista)
    
    if B < 10:
        if n == 1:
            return x_lista
        for d_i in x_lista:
            x_B.extend(division_reiterada(d_i, B))
    else:
        # sea 'r' la relacion entre B y 10. 
        # define cuantos digitos tomar como minimo para operar la div reiterada
        r = B // 10 + 1
        
        # sea n_r la cantidad de 
        n_r = n // r + 1

        for i in range (n_r):
            pass

    return x_B
"""