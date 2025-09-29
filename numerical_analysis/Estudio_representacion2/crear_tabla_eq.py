from PuntoFlotante_B import PuntoFlotante_B
import pandas as pd


""" Parte para leer desde el archivo """
with open( 'cp_conversion_16.txt' ) as archivillo:
    lineas = [linea.split() for linea in archivillo.readlines()] # Cada linea es una lista de palabras, la 1era es el nro

    """ Datos del algoritmo """
    B = 16 # base actual
    B_prima = 2 # base de llegada
    t = 52 # cantidad de digitos usados
    t_redondear = 5 # cantidad de digitos para redondear
    
    """ Datos para cargar la tabla"""
    datos = []
    nro_fila = 1

    """ Algoritmo """
    for nro_str in lineas:
        x_16 = PuntoFlotante_B(nro_str[0], B)

        if not x_16._chequear_errores(True):# true es para que muestre el tipo de error
            x_2 = x_16.convertir_a_base(B_prima, t)
            
            # Enves de mostrar, lo cargamos a un pandas
            datos.append( [nro_fila, x_16.mostrar_normalizado(), x_2.mostrar_sin_normalizar()] )
        
        nro_fila += 1
    
    """ Creo la tabla """
    df = pd.DataFrame(datos, columns=[
        'Nro de fila',
        f'Punto normalizado en base: {B}',
        f'Equivalente en base: {B_prima}'
    ])
    df.to_excel('tabla_conversion.xlsx')
