import math
import sys
def calcula_norma(x:list):
    """
    Calcula la norma de un vector.
    Tiene en cuenta problemas de representacion.
    Over/Under Flow.
    Suma ni Resta catrastofica ocurren
    """

    #acumuladores
    sum_pequenos=0
    sum_medianos=0
    sum_grandes=0

    #variables de control
    n=cota_pequenos=1.0e-150
    M=cota_grandes=1.0e+150
    num_max=sys.float_info.max
    num_min=sys.float_info.min
    
    #determinar los cuadrados de las coordenadas de un vector
    for x_i in x:
        if x_i < cota_pequenos:
            sum_pequenos+= (x_i*M)**2
        elif x_i > cota_grandes:
            sum_grandes+= (x_i*n)**2
        else:
            sum_medianos+= x_i**2
    
    #chequeamos si el overflow es inevitable
    if sum_grandes/n > num_max*n and sum_grandes!=0:
        print(f"Error desbordamiento en: {x}")
        return math.inf
    
    #chequeamos si el underflow es inevitable
    if sum_pequenos/M < num_min*M and sum_pequenos!=0:
        print(f"Error subdesbordamiento en: {sum_pequenos/n , num_min* M}")
        return 0.0

    sum= sum_medianos+ sum_pequenos/(M**2) + sum_grandes/(n**2)
    return math.sqrt( sum )

print(calcula_norma([1.0e+208,1,1.0e-188,1.0e-258,1.0e+158,1.0e-158,]))
print(calcula_norma([1.0e+108,1,1.0e-138,1.0e-158,1.0e+68,1.0e-48,]))
print(calcula_norma([1.0e+108,1,1.0e138,1.0e151,1.0e+68,1.0e-48,]))

