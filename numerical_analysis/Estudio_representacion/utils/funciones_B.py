"""
Módulo de funciones auxiliares para operaciones en base B
funciones que saque porque no aportan confunden
"""
def reestablecer_comas(self) -> None:
        def actualiza_coma_inv(b: bool) -> None:
            if b:
                self.pos_coma_inv = calcula_coma_inv(self.m_decimal)
        def actualiza_coma_dec(b: bool) -> None:
            if b:
                self.pos_coma = 1

def limpia(self) -> None:
        #caso m invertido [d0,d1,...,dn,COMA,e1,e2] -> [COMA, 0]
        self.m_inv = [0]
        self.pos_coma_inv = 0 + 1 # desde 0 hasta 1(excluido) son decimales
        
        #caso m no invertido [e1,e2,COMA,dn,...,d1,d0] -> [0,COMA]
        self.m_decimal = [0] 
        self.pos_coma = 1 # desde 1 hasta final son decimales    

def calcula_coma():
    return 1
def calcula_coma_inv(m_decimal: list):
    return len(m_decimal) 



