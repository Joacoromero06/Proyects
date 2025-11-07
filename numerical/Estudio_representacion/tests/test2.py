import sys
def muestra_path():
    for r in sys.path:
        print(r)

import os
ruta_actual = os.path.abspath(__file__)
directorio_ruta_actual = os.path.dirname(ruta_actual)

# EJemplo de join
otra_ruta = os.path.join('/home', 'GitHub', 'Proyects')
#print(otra_ruta)

# Añadir directorios al path
def setup_path():
    """
    Descripcion:
        Añadire los directorios necesarios al path, en este caso
        la raiz de proyecto 'Estudio_representacion'
    """
    ruta_actual = os.path.abspath(__file__)
    directorio_actual = os.path.dirname(ruta_actual)
    raiz_proyecto = os.path.dirname(directorio_actual)

    if raiz_proyecto not in sys.path:
        # Añade al inicio para maxima prioridad
        sys.path.insert(0, raiz_proyecto)

    muestra_path()
setup_path()
 