import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from utils.funciones import *
from abc import ABC
from models.Mantisa import MantisaDecimal, MantisaDecimal_10
from models.Maya import Maya

class PuntoFlotante(ABC):
    def __init__(self) -> None:
        self.s: int
        self.m: 'MantisaDecimal'
        self.e: int
        self.x_python: float
        self.maya: 'Maya'
    def __str__(self) -> str:
        s: str = '\n----PuntoFlotante----\n'
        s += f'SIGNO: {self.s}\n'
        s += f'MANTISA: 0.{list_dig2int(self.m.m_decimal)}\n'
        s += f'EXPONENTE: {self.e}\n'
        s += f'SU REPRESENTANTE EN PYTHON: {self.x_python}\n'
        s += f'\n----La mantisa esta respaldada por el objeto MantisaDecimal----\n{self.m}'
        return s
    def get_mantisa_entera(self) -> list:
        """
        Descripcion:
            self tiene un atribto m, su mantisa con este metodo obtendre
            m_ent los digitos de m que forman parte de la parte entera del
            numero representado por self

            Lista vacia si no tiene parte decimal
        """
        return self.m.get_dig_enteros()
    def get_mantisa_decimal(self) -> list:
        """
        Descripcion:
            self tiene un atribto m, su mantisa con este metodo obtendre
            m_dec los digitos de m que forman parte de la parte decimal del
            numero representado por self
            
            Lista vacia si no tiene parte decimal
        """
        return self.get_mantisa_decimal()

class PuntoFlotantePython(PuntoFlotante):
    def __init__(self, x: float) -> None:
        """
        Descripcion:
            Construye el representante x en la maya IEEE.
        Input:
            x: tipo float ya se encuentra en dicha representacion.
            donde x = d1d2...dn,d1d2...dm, talque n + m = 52
        Output:
            None: no retorna algo en su nombre
            punto_flotante_IEEE: es mas manejable donde el objeto
            x_pf_ieee = [d1, d2, ..., dn, d1, d2, ...,dm].
        """       
        super().__init__()
        self.m: 'MantisaDecimal_10'
        def agregar_datos():
            F_python = Maya(10,10,-307,308)
            self.maya = F_python
            self.x_python = x
        def calcula_signo():
            self.s = 1 if x > 0 else 0
        def calcula_mantisa():
            """
            Descripcion:
                Computa dado el float x con su parte entera y decimal
                representado por la mantisa de python, una MantisaDecimal
                correspondiente a ese x
            Pasos:
                1. La parte entera de x moverla la cantidad de digitos de la parte decima
                2. De esta forma con int2list obtenemos la m_decimal del objeto
                3. calculamos el exponente RELACION posicion coma en el exponente
            """
            def calcula_exp():
                """
                Descripcion:
                    Para calcular el exponente del numero simplemente calculamos la
                    cantidad de digitos de la parte entera del numero, sin embargo en
                    este caso el exponente lo usamos en papel, en el objeto lo que 
                    nos interesa es la posicion de la coma, eponente mas 1 debido
                    a que el cero al principio de la mantisa si se tiene en cuenta
                    en la lista del objeto
                """
                return cant_dig(parte_entera(x))
                
            x_ent = parte_entera(x)
            x_dec, cant_ceros = parte_decimal(x)
            print(f'\nx_ent: {x_ent}')
            print(f'\nx_dec: {x_dec}')
            print(f'\ncant_ceros: {cant_ceros}')
            
            x_transformado = x_ent * 10**(cant_dig(x_dec)+cant_ceros)
            x_transformado = x_transformado + x_dec
            print(f'\nx_transformado: {x_transformado}')
            x_lista = int2list(x_transformado)
            
            if x_ent == 0:
                for i in range ( cant_ceros ):
                    x_lista.insert(0, 0)
            
            print(f'\nx_lista: {x_lista}')
            exp = calcula_exp()
            self.m = MantisaDecimal_10(x_lista, exp + 1, self.maya.t)
            self.e = self.m.e
            
        agregar_datos()
        calcula_signo()
        calcula_mantisa()
    def __str__(self) -> str:
        return '\n----PuntoFlotantePython----\n' + super().__str__() 
    def get_equivalente_B(self,B: int) -> PuntoFlotante:
        def get_lista_ent():
            rep_ent_B = self.m.division_reiterada(B)
            lista_dig_ent = rep_ent_B.get_dig_enteros()
            dig_usados_ent = len(lista_dig_ent)
            return (lista_dig_ent.copy(), dig_usados_ent)
        def get_lista_dec(dig_usados_ent: int) -> list:
            dig_restantes_dec = self.m.t - dig_usados_ent 
            rep_dec_B = self.m.multiplicacion_reiterada(B, dig_restantes_dec)
            return rep_dec_B.get_dig_dec().copy()
        def unir_listas_entYdec(l_ent: list, l_dec: list) -> list:
            return l_ent + l_dec
        def calcula_exp(lis_ent: list) -> int:
            return len(lis_ent)
        def calcula_t() -> int:
            return self.maya.t
        l_ent, cd_usados = get_lista_ent() 
        print(f'\nl_ent = {l_ent}\ncd_usados = {cd_usados}')
        l_dec = get_lista_dec(cd_usados)
        print(f'\nl_dec = {l_dec}')
        dig_mantisa = unir_listas_entYdec(l_ent , l_dec )
        exp = calcula_exp(l_ent)
        t = calcula_t()
        print(f'\ndif_mantisa = {dig_mantisa}\nexp = {exp}\nt = {t}')
        mantisa_rep_B = MantisaDecimal(dig_mantisa, exp, t, B )
        #print(f'\nmantisa_rep_b = {mantisa_rep_B}')
        return PuntoFlotante_B(mantisa_rep_B, self, B)

class PuntoFlotante_B(PuntoFlotante):
    """
    Este objeto con las caracteresticas de cualquier flotante, pero
    en una base B
    """
    def __init__(self, m_rep_B: 'MantisaDecimal', x_pf_py:'PuntoFlotantePython', B: int) -> None:
        def agregar_datos():
            
            self.x_python = x_pf_py.x_python
            self.maya = Maya(B, x_pf_py.maya.t, x_pf_py.maya.L, x_pf_py.maya.U)
            self.s = x_pf_py.s
            
            self.m = m_rep_B
            self.e = m_rep_B.e
            
        super().__init__()
        agregar_datos()

if __name__ == '__main__':
    print('Ejecutando desde el archivo')
    x = 0.001023
    x_pf_python = PuntoFlotantePython(x)
    print(x_pf_python)
    print(x_pf_python.get_equivalente_B(16))


#class punto_flotante_F(punto_flotante):
