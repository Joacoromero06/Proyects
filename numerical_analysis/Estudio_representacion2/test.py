from PuntoFlotante_B import PuntoFlotante_B
import time
 

if False:
    pass
else:
    with open ('cp_conversion_16.txt') as archivillo:
        
        # Guardo todas las casos en una lista
        casos_prueba = [linea.split() for linea in archivillo.readlines()]
        
        for linea in casos_prueba:
            try:
                x_ = linea[0]
            except:
                pass
            
            B = 16; B_prima = 2; cd_sign = 10
            print(f'\nConversion de base {B} a {B_prima}')
            x_B = PuntoFlotante_B(x_,B)
            x_B_prima = x_B.convertir_a_base(B_prima)  
            if not x_B._chequear_errores(False):    
                x_B.print_normalizado()
                x_B_prima.print_normalizado()
                print(f'Aplicando redondeo simetrico a {cd_sign} digitos es:')
                (x_B_prima.aplicar_redondeo_simetrico(10)).print_normalizado()
                
            time.sleep(5)

#print(x_pf.aplicar_corte(1))
#print(x)





