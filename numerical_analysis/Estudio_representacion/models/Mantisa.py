import copy as copy_libreria
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.funciones import *
from collections import deque

class MantisaDecimal():
    def __init__( self, m: list, e: int, t: int, B: int ) -> None:
        """
        Descripcion:
            Construye la MantisaDecimal, que representa la parte decimal del numero
            que es representado por alguna Mantisa de algun PuntoFlotante
            Representa los numeros decimales que usamos en la multiplicacion reiterada
        Input:
            igual
        Output:
            m_decimal:
                type: list
                La parte decima de la m, puede ser cero/vacia
                [0, d1, d2, ...,dn]
            m_inv:
                type: list
                La parte decima de la m invertida
                [dn, d1, d2, ...,dn]
            pos_coma:
                type: int
                La posicion de la coma implicita en m_decimal
            pos_coma_inv:
                type: int
                La posicion de la coma implicita en m_inv
        
        """
        def trunca_m(m: list):
            """
            Descripcion:
                COn frecuencia la longitud de m sera de 52 numeros, debido a que es
                lo que maneja python, por lo que nos interesa manejar la cantidad que 
                nosotros queramos con t. Dada un intento de mantisa muy grande
                lo acortaremos a lo que t permite representar (maya)
                Tambien esta m puede ser mayor a 52 numeros un numero ingresado por
                nosotros como list de python obligatoriamente (ya que python y su
                float no permite mas de 52) y con un t mu grande podemos manejar esas
                representaciones
            Pasos:
                Como m representa los digitos de un numero flotante del tipo
                m: 0.[digitos de m]. Entonces len(m) <= t para representar con 
                exactitud a m. En caso de que len(m) > t tendremos errores inherentes
                y guardaremos dicho error en cd_error
            """
            return m[:t]
        def calcula_m_decimal() -> list:
            """
            Descripcion:
                Dada una lista que representa la mantisa de un PuntoFlotante
                y su exponente, computamos lo que hariamos al pasar de la notacion
                Punto flotante Normalizada: 0.m x B^e
                Al numero expandido, moviendo la coma implicita por ahora e veces
            """
            if e > len(m) - 1:
            #recordemos que este objeto representa el numero flotante en PFNormalizado
                pass#return []
            if e <= 0:
                cpy = deque(m[:])
                cpy.extendleft([0]*(abs(e) + 1))
                return  list(cpy)
                # m es la lista de digitos decimales de algun numero flotante -> no tiene parte entera
            cpy = deque(m[:])
            cpy.extendleft([0])
            return list(cpy)
        def calcula_exp() -> int:
            #print(f'\n\nm = {m}')
            cd_ceros_decimales = 0
            for d_i in m:
                if d_i != 0:
                    break
                cd_ceros_decimales += 1
            return cd_ceros_decimales
        m = trunca_m(m)
        #print(f'x_lista luego de truncar es: {m}')
        self.m_decimal = calcula_m_decimal()
        #print(f'm_decimal luego de prepararlo: {self.m_decimal}')
        self.m_inv = (self.m_decimal)[::-1]
        self.pos_coma = 1
        self.pos_coma_inv = len(self.m_decimal)
        self.dig_manejados_dec = len(self.m_decimal) - self.pos_coma
        self.dig_manejados_ent = self.pos_coma
        self.base = B
        self.e = -calcula_exp()
        self.t = t
    
    def __str__(self) -> str:
        s = '\nMANTISA DECIMAL:'
        s += '\nm_dec = ' + str(self.m_decimal)
        s += '\npos_coma = ' + str(self.pos_coma)
        s += '\nm_inv = ' + str(self.m_inv)
        s += '\npos_coma_inv = ' + str(self.pos_coma_inv)
        s += '\ndig_manejados_dec = ' + str(self.dig_manejados_dec)
        s += '\ndig_manejados_ent = ' + str(self.dig_manejados_ent)
        s += '\nBase = ' + str(self.base)
        s += '\nexp = ' + str(self.e)
        s += '\nt_significativos = ' + str(self.t)
        return s
    def es_vacia(self):
        return self.m_decimal == []
    def es_entera(self):
        pass 
    def get_dig_enteros(self):
        return self.m_decimal[:self.pos_coma]
    def get_dig_dec(self):
        return self.m_decimal[self.pos_coma:]
    def copia(self)-> 'MantisaDecimal':
        return copy_libreria.deepcopy(self)
    def actualizar_comas(self, coma_inv: int, coma_dec: int) -> None:
        """
        Descripcion:
            Dada 2 cantidades determinamos cuanto mover las comas de las m_inv y
            m_dec respéctivamente
        """
        self.pos_coma_inv += coma_inv
        self.pos_coma += coma_dec
    def actualiza(self, b_inv: bool) -> None:
        """
        Descripcion:
            Dada una bandera determinamos que mantisa actualizar con cual valor
            Si en un procedimiento se modifico m_inv -> actualizamos m_decimal y
            viceversa si ocurre al reves
        """
        if b_inv:
            self.m_inv = self.m_decimal.copy()[::-1]
        else:
            self.m_decimal = self.m_inv.copy()[::-1]
    def actualiza_dig_manejados(self) -> None:
        #print('¿NO ACTUALIZO?')
        self.dig_manejados_dec = len(self.get_dig_dec())
        #print(type(self.get_dig_dec()), len(self.get_dig_dec()))
        self.dig_manejados_ent = len(self.get_dig_enteros())
    def unit_last_in_place(self) -> 'MantisaDecimal':
        ceros = self.copia() 
        ceros.m_decimal = [0] * (self.dig_manejados_dec + self.dig_manejados_ent)
        ceros.m_decimal[len(ceros.m_decimal)-1] = 1
        ceros.actualiza(True)
        return ceros
    def suma_ulp_B(self) -> 'MantisaDecimal':
        """
        Descripcion:
            Computa la suma de una MantisaDEcimal con la unidad mas chica
            para su mantisa
        Pasos:
            Copiamos la MantisaDecimal que hay que sumar, para efectuar la suma
            en el y retornarlo
            Recorremos los elementos del numero decimal desde los menos significativos
            hasta los mas significativos
                Sumamos d_i con 1
                Si d_i == B entonces suma.m_inv[i] = 0
                Si no, entonces suma.m_inv[i] = d_i + 1, over_flow = False break
        """
        suma = self.copia()
        over_flow = True
        for i in range( len(self.m_inv) ):
            d_i = self.m_inv[i]
            if d_i == self.base - 1:
                suma.m_inv[i] = 0
            else:
                suma.m_inv[i] = d_i + 1
                over_flow = False
                break
        if over_flow:
            # No pasara porque MantisaDecimal tiene un 0 al inicio del decimal
            pass
        
        suma.actualiza(False)
        # No es necesario actualizar nada mas
        return suma
    def get_mantisa_entera(self) -> 'MantisaDecimal':
        """
        Descripcion:
            Computa y retorna la MantisaDecimal rep_ent de un objeto del mismo tipo.
            Donde rep_ent es una MantisaDecimal que representa la parte entera del
            numero represantodo por MantisaDEcimal original
        Pasos:
            Obtener la lista de digitos de la parte entera del objeto self
            Crear una MantisaDecimal con esa lista de digitos, con un exponente 
            adecuado, obtenido gracias a la cantidad de digitos enteros.
            REcordemos que la MAntisaDecimal representa la hoja infinita de papel 
        """
        list_ent = self.get_dig_enteros()
        x_ent = list_dig2int(list_ent)
        cd = cant_dig(x_ent)
        return MantisaDecimal(list_ent, cd, self.t, self.base)
    def get_mantisa_decimal(self) -> 'MantisaDecimal':
        """
        Descripcion:
            Computa la MantisaDecimal de la parte decimal del numero que representa
            la MantisaDecimal actual
        Pasos:
            Obtener la lista de digitos decimales del objeto.
            El exponente sera 0 puesto que es la parte decimal
        """
        return MantisaDecimal(self.get_dig_dec(), 0, self.t, self.base)

class MantisaDecimal_10(MantisaDecimal):
    def __init__( self, m: list, e: int, t: int ) -> None:
        super().__init__(m,e,t,10)
    def copia(self)-> 'MantisaDecimal_10':
        """
        Descripcion:
            Metodo que copia un objeto con la libreria copy
        """
        return copy_libreria.deepcopy(self)
    def producto(self, dig: int) -> 'MantisaDecimal_10':
        prod = self.copia()
        acarreo = 0
        for i in range( len(prod.m_inv) ):
            di_x_dig = self.m_inv[i] * dig # en la ultima iter da 0, por que x es 0.d1d2..
            en_i = di_x_dig + acarreo
            prod.m_inv[i] = en_i % 10
            acarreo = en_i // 10 
        if acarreo != 0:
            print('EN PRODUCTO, LA PARTE ENTERA NO ERA: 0 overflow con: ',acarreo )

        prod.actualiza(False)
        return prod      
    def empareja(self, y:'MantisaDecimal_10'):
        """
        Descripcion:
            Empareja dos MantisaDecimales_10 al estiulo papel
        Pasos:
            Determinamos cuantos 0, de parte entera agregar
            Se los agregamos al que correspone como los mas significativos(son cero
            igual)
            Determinamos cuantos 0, de parte decima agregar
            Se los agregamos al que corresponde como los menos significativos( son cero
            no afecta el valor de todas formas)
            Actualizamos al fina los digitos manejados, rtdo de agregar ceros
        """
        dif_dig_ent = self.dig_manejados_ent - y.dig_manejados_ent
        if dif_dig_ent > 0:
            y.m_inv.extend([0]*dif_dig_ent)
            y.actualiza(False)
            y.actualizar_comas(0, dif_dig_ent)
        if dif_dig_ent < 0:
            self.m_inv.extend([0] * abs(dif_dig_ent))
            self.actualiza(False)
            self.actualizar_comas(0, abs(dif_dig_ent))

        dif_dig_dec = self.dig_manejados_dec - y.dig_manejados_dec
        if dif_dig_dec > 0:
            y.m_decimal.extend([0]*dif_dig_dec)
            y.actualiza(True)
            y.actualizar_comas(dif_dig_dec, 0)
        if dif_dig_dec < 0:
            self.m_decimal.extend([0]*abs(dif_dig_dec))
            self.actualiza(True)
            self.actualizar_comas(abs(dif_dig_dec), 0)
        self.actualiza_dig_manejados()
        y.actualiza_dig_manejados()
    def suma(self, y: 'MantisaDecimal_10') -> 'MantisaDecimal_10':
        """
        Descripcion:
            Dados dos MantisasDecimales_10 copmutamos la suma al estilo papel
        Pasos:
            Emparejamos con ceros para que no hay aproblemas
            En un nuevo objeto de la misma forma, la coma en la misma posicion.
            Vamos realizando la suma desde los dig menos significativos hasta los mas
            En cada cifra de en_i obtenemos el digito de la suma en esa cifra y el
            acarreo para la siguiente
            En caso de que haya acarreo, hay que agregar el acarreo como el dig
            mas siugnificativo del numero, ademas actualizar la coma para m_dec por uno a la
            der
        """
        self.empareja(y)
        """print(f'\nEMPAREJANDO NUMEROS\n')
        print(self)
        print(y)"""
        suma = self.inicializa_con_forma()
        acarreo = 0
        for i in range(len(suma.m_inv)):
            xi_mas_yi = self.m_inv[i] + y.m_inv[i]
            en_i = xi_mas_yi + acarreo
            suma.m_inv[i] = en_i % 10 
            acarreo = 1 if en_i >= 10 else 0
            #print(f'\nITERACION = {i}\n self.m_inv[i]: {self.m_inv[i]} + y.m_inv[i]: {y.m_inv[i]}+ acarreo: {acarreo}= en_i: {en_i}')
        if acarreo != 0:
            #print('\nEN SUMA, LAS PARTE ENTERAS NO ERAN: 0 DIERON: ',acarreo )
            suma.m_inv.append( acarreo )
            #mueve la coma de m_decimal uno para la derecha
            suma.actualizar_comas(0,1)       

        suma.actualiza(False)
        suma.actualiza_dig_manejados()
        return suma
    def inicializa_con_forma(self) -> 'MantisaDecimal_10':
        """
        Descripcion:
            Metodo para crear un objeto de las mismas cualidades del que lo llama
            pero solo ceros, util para efectuar sumas y emparejar
        Pasos:
            Copiamos todo con copia()
            m_decimal sera una lista de CEROS * len(m_decimal)
            Se actualiza m_inv, (lo copia en m_inv)
            Las comas quedaron bien puestas por la copia del inicio
        """       
        ceros = self.copia() 
        ceros.m_decimal = [0] * (self.dig_manejados_dec + self.dig_manejados_ent)
        ceros.actualiza(True)
        return ceros
    def mult_x_10(self) -> None:
        """
        Descripcion:
            Computa lo que realizamos al multiplicar por 10 un numero decimal
        Pasos:
            Se agrega un 0 al final que no representa volor por ser decimal
            Se actualiza el m_inv
            Se corre la coma, solo en el caso de la representacion de m_decimal
            Al correr la coma los digitos que maneja el objeto cambio
        """
        self.m_decimal.extend([0])
        self.actualiza(True)
        self.actualizar_comas(0,1)
        self.actualiza_dig_manejados()
    def mult_x_10_i_veces(self, i: int) -> None:
        for j in range(i):
            self.mult_x_10()
    def producto_B(self, B: int) -> 'MantisaDecimal_10':
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
        producto = self.inicializa_con_forma()
        cd = cant_dig(B)
        for i in range(cd):
            dig_actual = B % 10
            #print(f'\nPara el dig: {dig_actual}\n')

            prod_dig = self.producto(dig_actual)
            #print(f'\nprod dio:{prod_dig}')

            prod_dig.mult_x_10_i_veces(i)
            #print(f'\nLa suma entre: {producto.m_inv} y {prod_dig.m_inv}')
            
            producto = producto.suma( prod_dig )
            #print(f'\nSUMA:{producto.m_inv}')
            B = B // 10
        return producto
    def multiplicacion_reiterada(self, B: int, t_restante: int) -> 'MantisaDecimal':
        """
        Descripcion:
            COmputa la multiplicacion reiterada de una mantisaDecimal, un decimal
            Con un numero B que representa la nueva base.
        Pasos:
            Realizar a lo sumo t veces:
                1.Multiplicar el producto_actual con B
                La parte entera es un nuevo digito del representante_B de tipo
                2.MantisaDecimal, se lo obtiene, con get_dig_ent() y con list_2_int
                producto_actual es la parte decimal de si mismo, para eso
                3.Eliminamos los digitos para m_dec, corregimos la coma, actualizamos
                m_inv y los digitos que maneja el objeto
                4.Si producto_actual no tiene parte decimal, entonces termino
            Si no se termino:
                Seguimos con el procedimiento, sin agregar a representante_B
                y buscando un tier_breaking, para determinar si hay redondeo
                Esta parte depende de si la base es par o impar
            Si hay redondeo
                generamos unit_last_in_place para B y sumamos con representante_B
        """
        m_decimal_aux = []
        p_actual = self.copia()
        tier_breaking = False
        redondeo = False
        print(f'\nINICIO\nrepresentante_B: {m_decimal_aux}')


        for i in range(t_restante):
            p_actual = p_actual.producto_B(B)
            print(f'\np_actual: {p_actual}\n\n\n')
            dig_B = list_dig2int(p_actual.get_dig_enteros())
            m_decimal_aux.append(dig_B)
            
            if list_dig2int(p_actual.get_dig_dec()) == 0:
                tier_breaking = True
                break
            p_actual = MantisaDecimal_10(p_actual.get_dig_dec(), -1, self.t)
            print(f'Rtdo de sacarles los dig decimales:{p_actual}')
        while not tier_breaking:
            print('\nTIER BREAKING')
            p_actual = p_actual.producto_B(B)
            dig_B = list_dig2int(p_actual.get_dig_enteros())
            print(f'\nRedondear? {dig_B}')
            if B % 2 == 0:
                if dig_B >= B // 2:
                    redondeo = True
                else:
                    redondeo = False
                break
            else:
                if dig_B >= B // 2 + 1:
                    redondeo = True
                    break
                elif dig_B < B // 2 + 1: 
                    redondeo = False
                    break
                print('hay ciclo infinito?')
        representant_B = MantisaDecimal(m_decimal_aux,0,0,B)
        representant_B.actualiza(True)
        representant_B.actualiza_dig_manejados()
        representant_B.actualizar_comas(len(representant_B.m_decimal)-1-1,0)
        if redondeo:
            print('Hay que redondea')
            representant_B = representant_B.suma_ulp_B()
        
        return representant_B
    def division_reiterada(self, B: int) -> 'MantisaDecimal':
        x_ent = list_dig2int(self.get_dig_enteros())
        m = deque()
        while x_ent >= B:
            dig = x_ent % B
            x_ent = x_ent // B
            m.appendleft(dig)

        representante_B = MantisaDecimal(list(m), len(m)-1, 0, 10)
        representante_B.actualizar_comas(-len(m), len(m) )
        return representante_B

if __name__ == '__main__':
    print('\nDATOS')
    m = MantisaDecimal_10(   [2], 0, 6);print(m)
    #m1 = MantisaDecimal_10([6,7,8,9], -2, 6); print(m1)
    print(m.multiplicacion_reiterada(3,6))
    
    #print('OPERACIONES')
    #B = 16
    #m_x_B = m.producto_B(B); print(m_x_B)
    """print('OPERACIONES')
    d = 6
    m_x_d = m.producto(d)
    print(f'\nEl producto es:{m_x_d}')

    m_mas_m1 = m.suma(m1)
    print(f'\nLa suma es{m_mas_m1}')"""