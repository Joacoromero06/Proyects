import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.funciones import *
from models.punto_flotante import punto_flotante_10

def main1():
    x = 2.31
    x_dec = parte_decimal(x)
    x_lista = int2list(x_dec)
    print(f'la parte decimal de {x} es {x_dec} \n')
    print(f'la cantidad de dig es {cant_dig(x_dec)}')
    
    x = 3.14
    x_dec = parte_decimal(x)
    x_lista = int2list(parte_decimal(x))
    print(f'la mantisa de {x} \nEs: {x_lista}')
    print(f'la cantidad de dig es {len(x_lista)}')
    
def main2():
    x_pf: punto_flotante_10 = punto_flotante_10(221.31)
    print(x_pf)
    print(x_pf.get_mantisa_entera())
    print(x_pf.get_mantisa_decimal())

main2()