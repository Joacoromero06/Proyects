import math
from typing import List, Union
class PuntoFlotante_B:
    def __init__(self, numero_str: str, base: int):
        """
        Constructor que recibe el número como string y la base.
        Ejemplo: PuntoFlotante_B("123.45", 10) o PuntoFlotante_B("-101.011", 2)
        """
        
        # Clasica inicializacion de un objeto
        self.base = base
        self.signo = '+'  # 1 para positivo, -1 para negativo
        self.mantisa_entera: List[Union[int,str]] = []  # Lista de dígitos de la parte entera
        self.mantisa_decimal: List[Union[int,str]] = []  # Lista de dígitos de la parte decimal
        self.exponente = 0  # Se calculará según las reglas que definiste
        self.error = 0
        self.es_decimal = True

        # Genera un conjunto con letras validas para los numeros de la base
        def generar_letras_validas():
            return [chr(c) for c in range(65, 65 + self.base -10, 1)]
        self.letras = generar_letras_validas()
        self._parsear_numero(numero_str)
        if self._chequear_errores():
            self._chequear_errores(True)
            self.mantisa_decimal = []
            self.mantisa_entera = []
            print('Object Error')
        else:
            self._calcular_exponente()
        
    def _parsear_numero(self, numero_str: str):
        """
        Método privado que parsea el string y llena mantisa_entera y mantisa_decimal
        """
        # Determina si un string de un caracter es un digito o una letra del dominio
        def in_dominio(d) -> bool:
            if self.base <= 10:
                digitos = [str(i) for i in range(0, self.base)]
            else:
                digitos = [str(i) for i in range(0, 10)]
            #print(f'¿c: {d, type(d)} esta en el dominio? : {self.letras} o {digitos}')
            if d in self.letras or d in digitos:
                return True
            else:
                return False
        def controles_refinar_str(self, numero_str: str) -> str:
            """
            Descripcion:
                Controla que no sea str vacio -> err100
                COntrola que los digitos sean de la base ->err300
                Refina espacios en blanco
                Refina signo, y lo saca del str
                Refina los 0 sin valor ¿saca el 0 del . ej: 00.01 -> .1?
            """
            
            # Contralando posible falla -> No ingresa nada
            if not numero_str or numero_str == '':
                print('llego un str vacio')
                self.error = 100
                return ''
            
            # Controlando digitos que no pertenezcan a la base
            for c in numero_str:
                if c == '.' or c == ',' or c == '+' or c == '-':
                    continue 
                if not in_dominio(c):
                    #print(f'el caracter c: {c} no esta el dominio -> error 300 retornamos str vacia')
                    self.error = 300
                    return ''

            # Consumir los espacios en blanco
            while numero_str[0] == ' ':
                numero_str = numero_str.removeprefix(' ')
                if not numero_str:
                    break

            # Determinar signo y sacar el signo para permitir parsear
            c_mas = 0; c_menos = 0
            while numero_str[0] == '+' or numero_str[0] == '-':
                if numero_str[0] == '+':
                    c_mas += 1 
                    numero_str = numero_str.removeprefix('+')
                else:
                    c_menos += 1
                    numero_str = numero_str.removeprefix('-')
            if c_menos % 2 == 0:
                self.signo = '+'
            else:
                self.signo = '-'    
        
            
            # Remueve todos los 0s sin valor que el usuario puede poner, ej: 000123.2
            while numero_str[0] == '0':
                numero_str = numero_str.removeprefix('0')
                if not numero_str: 
                    break
        
            return numero_str
        
        # Abortamos si hubo errores sino, se filtro correctamente
        numero_str = controles_refinar_str(self, numero_str)
        if self._chequear_errores():
            #print(f'como el error es: {self.error} en _parsear_numero retornamos: None')
            return
        
        # Bandera que determina que parte del numero estamos cargando
        cargar_enteros = True

        # Solo muestro si no es el objeto aux_err: '0.0'
        if numero_str != '.0':
            pass
            #print(f'String que se va a parsear: {numero_str}')# ete e el famoso
        
        # Calcula la lista de dig enteros y decimales
        ya_vio_coma = False
        for d in numero_str:
            
            # SI es coma -> termino parte entera, no puede haber otra coma
            if d == '.' or d == ',':
                if ya_vio_coma:
                    self.error = 200
                    return
                    
                cargar_enteros = False
                ya_vio_coma = True
            
            # Si no chequeo que sea del dominio el digito
            else:
                if not in_dominio(d):
                    self.error = 300
                    return
                    
                
                # Si es un digito, lo cargo como int
                if d.isdigit():
                    
                    # Si no vimos coma -> estamos cargando parte entera
                    if cargar_enteros:
                        self.mantisa_entera.append(int(d)) 
                    else:
                        self.mantisa_decimal.append(int(d))
                
                # Si es un caracter como 'A' -> lo cargo como str
                else:

                    # Si no vimos coma -> estamos cargando  parte entera
                    if cargar_enteros:
                        self.mantisa_entera.append(d) 
                    else:
                        self.mantisa_decimal.append(d)    
        
        # Agrega 0 adelante de la coma
        if self.mantisa_entera == []:
            """
            Esto se debe a que por eliminar los 0s sin valor de adelante
            Podemos eliminar el 0 de un decimal ej: 0,d1..dn -> ,d1..dn 
            """
            self.mantisa_entera = [0]
          
        # Si no habia punto ni coma en el str
        if cargar_enteros:
            #print('porque entra aca ??')
            self.error = 400
            self.es_decimal = False
        
        
        # Si hay coma y decimales
        else:
            # Elimino los 0s sin valor decimales
            while self.mantisa_decimal and self.mantisa_decimal[-1] == 0:
                self.mantisa_decimal.pop()
            self.mantisa_decimal = self.mantisa_decimal if self.mantisa_decimal else []
            
            # Si si hay coma, pero no decimales ej: d1..dn,nada             
            if not self.mantisa_decimal:
                self.error = 410
                self.es_decimal = False
    def _calcular_exponente(self):
        """
        Calcula el exponente según la sig lógica:
        - Si mantisa_entera tiene longitud 1 y es [0], cuenta ceros iniciales en mantisa_decimal
        - Si no, exponente = longitud de mantisa_entera
        """
        # Si hay error no calculo nada
        if self._chequear_errores(False):
            return
        
        #print('\nEstamos calculando exponente')
        # Si el unico digito entero del numero es 0, entonces el numero
        # puede que tenga exponente nulo o negativo
        if self.mantisa_entera[0] == 0 and len(self.mantisa_entera) == 1:
            c = 0
            for d in self.mantisa_decimal:
                if d == 0:
                    c += 1
                else:
                    break
            #print(f'devuelve la cant de ceros decimales consecutivos: {c}')    
            self.exponente = -c

        # Sino entonces el exponente esta dado por la cantidad de digitos enteros
        else:
            #print(f'Le asigna al exponente la longitud de la sig lista de dig enteros: {self.mantisa_entera}')
            self.exponente = len(self.mantisa_entera)
            #print(f'Entonces el exponente del objeto es:{self.exponente}')
    def _chequear_errores(self, mostrar = False):
        e = self.error
        error = False
        if e > 0 and e <= 300:
            error = True
        if mostrar:
            if e == 0:
                print('No hay errores')
            elif e > 0 and e <= 300:
                error = True#####
                if e == 100:
                    print('ERROR 100: No ingreso numero')
                if e == 200:
                    print('ERROR 200: Formato invalido -> Doble coma ')
                if e == 300:
                    print('ERROR 300: Caracter invalido')
            elif e >= 400:
                if e == 400:
                    print('ADVERTENCIA 400: Se encontro un entero')
                if e == 410:        
                    print('ADVERTENCIA 410: Se encontro un punto al final') 

        return error 

    def mostrar_sin_normalizar(self) -> str:
        """
        Muestra el número en su forma original sin normalizar.
        Ejemplo: 123.45 en base 10
        """
        # Mostramos solo si no hay errores
        
        if self._chequear_errores(False):
            return 'Error al mostrar_sin_normalizar()'
        
        s = f'{self.signo}'
        for d in self.mantisa_entera:
            s += str(d)
        s += '.'
        for d in self.mantisa_decimal:
            s+= str(d)
        return s  
    def mostrar_normalizado(self) -> str: 
        """
        Muestra el número en forma normalizada.
        Ejemplo: 0.12345 x 10^3 para 123.45 en base 10
        """
        # Mostramos solo si no hay errores
        if self._chequear_errores(False):
            return 'Error al intentar mostrar_normalizado()'
        
        s = f'{self.signo}0.'

        # En caso de que el numero no tenga parte entera
        if self.mantisa_entera == [0]: 
            
            # Si no tiene parte decimal
            if not self.es_decimal :
                pass
            else:
                i = 0
                # Evito los ceros luego de la coma sin significancia para el punto flot
                while i < len(self.mantisa_decimal) and self.mantisa_decimal[i] == 0:
                    i += 1
                
                # Agregos los digitos, son todos significantes 
                while i < len(self.mantisa_decimal):
                    # _parsear_ elimina los ceros del final
                    s += str(self.mantisa_decimal[i])
                    i += 1
                
        # Si tiene parte entera
        else:
            # Si tiene parte decimal, muestro todo
            if self.es_decimal:
                # Recordemos que _pasear_ refiina matisa_decimal
                for d in self.mantisa_entera + self.mantisa_decimal:
                    s += str(d)
            
            # Si no tiene, chequeo los ceros ej: 2000 = 2xB³ 
            else:
                # no muestro los ceros insgnificantes debida a xBase^exp
                
                # Determino la posicion del ultimo dig significante
                ult_significante = 0
                i = 0
                while i < len(self.mantisa_entera):
                    if self.mantisa_entera[i] != 0:
                        ult_significante = i
                    i += 1
                
                # La lista de digitos a añadir a s (str) son hasta ult_significante
                i = 0
                while i <= ult_significante:
                    s += str(self.mantisa_entera[i])
                    i += 1
                    
        # Terminamos mostrandolo debidamentes
        s += f' x {self.base} ^ {self.exponente}'
        return s
    def print_normalizado(self):
        print(self.mostrar_normalizado())
    def print_sin_normalizar(self):
        print(self.mostrar_sin_normalizar())

    def aplicar_corte(self, t):
        """
        Retorna una nueva instancia aplicando corte a t dígitos significativos.
        """
        # SI hay errores, salgo
        if self._chequear_errores():
            aux_err = PuntoFlotante_B('0.0', self.base)
            aux_err.error = self.error
            return aux_err
        
        # Si tiene parte decimal
        if self.es_decimal:
            # Obtengo una lista con cada elemento un caracter del str
            s = self.mostrar_sin_normalizar()
            s = s.removeprefix('+')
            s = s.removeprefix('-')
            while s[0] == '0':
                s = s.removeprefix('0')
            x = list(s)
        
            # Obtengo la posicion en la lista donde esta la coma o punto
            pos_ent = 0
            try:
                pos_ent = x.index('.')
            except:
                pass
            try:
                pos_ent = x.index(',')
            except:
                pass
            
            # Saco la coma o punto de la lista
            try:
                x.remove('.')
            except:
                pass
            try:
                x.remove(',')
            except:
                pass

            # Si no es un numero sin parte entera
            if pos_ent != 0:
                # Trunco la lista con slices, si t es mayor que lista devuelve len(list)
                x_truncado = x[:t]
                
                # Si la cant_significativos es mas chica que la cant de enteros
                if t < pos_ent:

                    # Agrego los ceros que representan los digitos enteros
                    x_truncado.extend( ['0'] * (pos_ent - t) )
                
                # Si no, tengo que agregar la coma en la lista
                else:
                    x_truncado.insert(pos_ent, '.')
                
                # Vuelvo la lista de digitos str a un str
                x_truncado = ''.join(x_truncado)

                return PuntoFlotante_B(x_truncado, self.base)

            # Si no tiene parte entera -> chiquito
            else:
                
                # No cuento los ceros al lado de la coma
                i = 0        
                while x[i] == '0':
                    i += 1
                
                # En i quedo el 1er dig significante
                x = x[: i+t] # desde ahi agarro los t significantes
                
                # Agrego la coma
                x.insert(pos_ent, '.')

                # Retorno
                s = ''.join(x)
                return PuntoFlotante_B(s, self.base)

        else:
            # Trunco los digitos significativos
            x_truncado = self.mantisa_entera[:t]

            # Si la cant de digitos significativos es menor a la de enteros
            if t < len(self.mantisa_entera) :
                #print(f'¿no extiende? con { ( len(self.mantisa_entera) - t )}0s')
                x_truncado.extend( [0] * ( len(self.mantisa_entera) - t ) )
            
            # Si no, el x_truncado es el correcto por como es el slice en python
            else: 
                pass
                
            # Como x_truncado es una lista de int, la llevo a lista de str's
            x_truncado = [str(d) for d in x_truncado]

            # Como x_truncado es una lista de str's (cada uno un digito) uso join
            x_truncado = ''.join(x_truncado)
            
            # Armo el correspondiente objeto con el string armado
            return PuntoFlotante_B(x_truncado, self.base)
    def aplicar_redondeo_simetrico(self, t):
        """
        Retorna una nueva instancia aplicando redondeo simétrico a t dígitos significativos.
        """
        
        # Obtener una lista con los digitos enteros el punto y los decimales
     
        x = self.mantisa_entera + [math.inf] + self.mantisa_decimal

        def det_redondear_pos_caso():
            
            # Variable a determinar
            redondear = None

            # Determinamos desde donde buscar el tier breaking
            pos = -1
            c_pasos = -1 if self.mantisa_entera == [0] else 0
            while c_pasos <= t-1 and pos < len(x):
                if x[pos] == math.inf:
                    pos += 1
                    continue
                pos += 1
                c_pasos += 1

            # Determinamos en que caso estamos
            if t <= len(x) - 1:
                caso = 1 # caso basico: d1d2...dn,c1c2...cm
                if not self.mantisa_decimal:
                    caso = 2 # caso no hay decimales: d1d2...dn,
            else:
                caso = 3 # caso t es mayor que la cantidad de digitios

            # Desde la posicion buscamos el tier breaking
            i = pos + 1 
            while i < len(x) and redondear == None:
                if x[i] == math.inf:
                    i += 1
                    continue
                
                # Llevamos alfa -> int
                if type(x[i]) == str:
                    x_i = ( ord( str( x[i] ) ) - 65 ) + 10
                else:
                    x_i = x[i]
                
                if self.base % 2 == 0:
                    if int(x_i) >= self.base // 2:
                        redondear = True
                    else:
                        redondear = False
                else:
                    if int(x_i) > self.base // 2 + 1:
                        redondear = True
                    elif int(x_i) < self.base // 2 + 1:
                        redondear = False
                    else: 
                        i += 1

            # En caso que no se consiguio un tierbreaking, redondeamos
            if i == len(x):
                redondear = True
            
            return (redondear, pos, caso)

        # Determinamos si hay que redondear, la pos de t dig significativos en x, y el caso en el que nos llego x
        redondear,pos,caso = det_redondear_pos_caso()
        
        # Si hay que redondear
        if redondear:
            print(f'VA A REDONDEAR') 
            # Desde la posicion de t digitos significativos, redondeamos para arriba
            i = pos 
            while i >= 0:      
                # Si es coma la pasamos
                if x[i] == math.inf:
                    i -= 1
                    continue

                # Si la base es solo de digitos
                if self.base <= 10:
                    
                    # Si es el digito mas grande acarreo
                    if x[i] == self.base - 1:
                        x[i] = 0
                        i -= 1
            
                    # Si no es el digito mas grande  lo incrementamos
                    else:
                        x[i] = int (x[i]) + 1 
                        break
        
                # Si la base es alfanumerica
                else:
            
                    # Si es la letra mas grande acarreo
                    letra_mas_grande = self.letras[-1]
                    if x[i] == letra_mas_grande:
                        x[i] = 0
                        i -= 1
            
                    # Si no, incremento acorde el type
                    else:
                        # si es una letra incremento
                        if type(x[i]) == str:
                            c = str(x[i])
                            x[i] = chr( ord( c ) + 1 )
                
                        # Si es 9 pasa a ser 'A'
                        elif x[i] == 9:
                            x[i] = 'A'
                        
                        # Si es menor que 9 incremento
                        else:
                            x[i] = int(x[i]) + 1
                        
                        # SALGO
                        break                 
    
            # Si quedo acarreo, y el numero no era chiquito
            
            if i == -1 and not self.mantisa_entera == [0]: 
                # Agregamos un uno al inicio
                x.insert(0, 1)
    

            # Armamos una lista de str, con cada digito
            x_str = [str(d) if d != math.inf else '.' for d in x ]

            # Unimos todos los strings de digitos y un punto en uno solo
            s = ''.join(x_str)

            # Con el str armado, creamos un objeto redondeado simetricamente
            aux = PuntoFlotante_B(s, self.base)# probar con 0.FFF
            return aux.aplicar_corte(t)

        # Si no hay que redondear, hay que truncar    
        else:
            #print('trunca nomas')
            return self.aplicar_corte(t)
    
    def convertir_a_base(self, nueva_base, cd_dig=52):
        """
        Convierte este número a una nueva base usando float como intermediario.
        Retorna una nueva instancia de PuntoFlotante_B en la nueva base.
        """
        def multplicacion_r(t, decimales = self.mantisa_decimal, B = nueva_base) -> str:
            
            # Convierto la lista de dig decimales a un float 0,d1...dn
            x = 0
            exp = -1
            for d in decimales:
                x += d * 10 ** exp
                exp -= 1
            
            # Si es 0 en cualquier base tambien
            if x == 0:
                decimales_B = [0]; i = None; historial = None
            else:

                # Algoritmo multiplicacion reiterada
                i = 0
                decimales_B = []
                historial = set()
                while x != 0 and i < t:# seguramente x nunca sea 0
                    # Si el numero decimal ya aparecio, hay un patron
                    if x in historial:
                        print('patron encontrado')
                        break
                    if x == 0:
                        print('se redujo a 0 la parte decimal')
                        break
                    
                    # Agrego al historia
                    historial.add(x)

                    # Multiplico reiteradamente
                    x *= B
                    nvo_dig = int(x)

                    # Si representa un alfa -> lo convierto
                    if nvo_dig >= 10:
                        nvo_dig = chr( 65 + (nvo_dig - 10) )
                    decimales_B.append(nvo_dig)

                    # Saco la parte entera
                    x -= int(x)
                    i += 1
            
            # Lo vuelvo listas de str -> luego todo str
            decimales_B = [str(d) for d in decimales_B]
            return ''. join(decimales_B)
        def division_r(enteros = self.mantisa_entera, B = nueva_base) -> str:

            # Convierto la lista de enteros en un int
            x: int = 0
            for d in enteros:
                x *= 10
                x += d
            
            # Si es 0 -> en cualquier base tambien
            if x == 0:
                enteros_B = [0]
            else:
                # Algoritmo de division reiterada
                enteros_B = []
                while x != 0: 
                    # Si el representante es alfa -> convierto
                    nvo_dig = x % B
                    if nvo_dig >= 10:
                        nvo_dig = chr( 65 + (nvo_dig - 10) )
                    enteros_B.append(nvo_dig)
                    
                    # Divido reiteradamente
                    x = x // B
                    
            # Doy vuelta por que va al reves    
            enteros_B = enteros_B[::-1]

            # Convierto a lista de str -> luego a un str
            enteros_B = [str(d) for d in enteros_B]
            return ''.join(enteros_B)
        def suma_ponderada(enteros = self.mantisa_entera, decimales = self.mantisa_decimal, base_actual = self.base) -> 'PuntoFlotante_B':
                    
            # Ejecutamos suma ponderada a los enteros
            x_ent: int = 0
            i = 1
            for d in enteros:
                d_i = d if type(d) == int else  ord( d ) - 65 + 10
                x_ent += d_i * base_actual ** (len(enteros) - i )
                i += 1
            
            # Ejecutamos suma ponderada a los decimales
            x_dec: float = 0
            e = -1
            for d in decimales:
                d_i = d if type(d) == int else  ord( d ) - 65 + 10
                x_dec += d_i * base_actual ** e
                e -= 1
            
            s2 = str(x_dec)
            s2 = s2.removeprefix('0.')
            # Creamos un objeto base 10 equivalente
            return PuntoFlotante_B( str(x_ent)+'.'+s2, 10 )

        if self.base != 10:
            eq_10 = suma_ponderada()
            # print(eq_10)
            # print(eq_10.mostrar_normalizado())
            eq_B = eq_10.convertir_a_base(nueva_base, cd_dig)
            return eq_B
        else:
            s_ent = division_r()
            cd_restante = cd_dig - len(s_ent)
            # print(f'cd_res: {cd_restante}, cd_dig: {cd_dig}, len(s_ent):{len(s_ent)}')
            s_dec = multplicacion_r(cd_restante)
            return PuntoFlotante_B(s_ent+'.'+s_dec, nueva_base)
    
    def _a_float(self):
        """
        Convierte la representación interna a float de Python.
        Usa suma ponderada: suma(digito * base^posicion)
        """
        # TODO: Implementar conversión a float
        pass
    def _desde_float(self, valor_float, nueva_base):
        """
        Crea una nueva representación desde un float hacia nueva_base.
        """
        # TODO: Implementar conversión desde float
        pass
    def __str__(self):
        """
        Representación del número para debuggear
        """
        if 0 < self.error < 400:
            return f'objeto nulo, por error: {self.error}'
        return f"Signo: {self.signo}\nBase: {self.base}\nExponente: {self.exponente}\nEntero: {self.mantisa_entera}\nDecimal: {self.mantisa_decimal}"