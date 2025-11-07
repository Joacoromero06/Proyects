from __future__ import annotations
from Coefficient import Coefficient, IntegerCoefficient, RationalCoefficient, RealCoefficient
from typing import Tuple, List
import copy

class Polynomial:
    def __init__(self, coeffs: List[Coefficient]) -> None:
        """ Inicializacion del objeto
        Descripcion: 
            La List[Coefficient] coincide sus posiciones en la lista con 
            el nro de exponente de la variable a la que multiplica
        Ejemplo:
            [3.14, 1/3, 2, 1/3] --map--> 3.14 + 1/3 x + 2 x² + 1/3 x³
        Descripcion:
            el _grade del Polynomial corresponde siempre al tamaño de List[Coefficient]
            En python type list empieza en cero, mientras que la funcion len()
            computa el tamaño de la lista
            El grado de un Polynomial es len(self.coffs) - 1
        Ejemplo:
            [3.14, 1/3, 2, 1/3] --(len()-1)-->  3
        """
        self._coeffs: List[Coefficient] = coeffs
        self._grade: int = len(self.coeffs) - 1
    def __mul__(self, a: Coefficient) -> Polynomial:
        """
        Descripcion:
            Efectua producto entre un escalar a y el Polynomial self
        """
        coeffs: List[Coefficient] = []
        for c in self.coeffs:
            coeffs.append(c * a)
        return Polynomial(coeffs)

    @property
    def coeffs(self) -> List[Coefficient]:
        return self._coeffs
    
    @property
    def grade(self) -> int:
        return self._grade

    def horner1(self, x: float) -> Tuple[Polynomial, float]:
        x_0 = RealCoefficient(x)
        b_i = RealCoefficient(0)
        coeffs: List[Coefficient] = []
        for a_i in self.coeffs[::-1]:
            b_i = a_i + b_i*x_0
            coeffs.append(b_i)
        n = self.grade
        coc = coeffs[:n]
        resto = coeffs[n].value
        return Polynomial(coc[::-1]), resto
    def div_1(self, Q: Polynomial) -> Tuple[Polynomial, float]:
        """     
        Descripcion:
            Computa la division entre el Polynomial self con el Polynomial q
            Donde q es de la forma x (+ | -) a
        """
        # Computar solo solo si Q tiene grado 1
        if Q.grade != 1 and not Q.es_monico():
            raise ValueError(f'q: {Q} debe ser de la forma x (+ | -) a')
        a = Q.coeffs[0] 
        return self.horner1(-a.value)
    def div_2(self, Q: Polynomial) -> Tuple[Polynomial, float]:
        """
        Descripcion:
            Computa la division entre el Polynomial self con el Polynomial Q
            Donde Q es de la forma ax (+ | -) b
        """
        if Q.grade != 1 and not Q.es_monico():
            raise ValueError(f'q: {Q} debe ser de la forma x (+ | -) a')
        
        # No modificar los datos de Q
        q = copy.deepcopy(Q)

        # Dividir Q por a
        b, a = q.coeffs[0], q.coeffs[1]
        q.coeffs[1] = a/a
        q.coeffs[0] = b/a 

        # Computar division: self / x (+ | -) b/a
        C_prima, r_prima = self.div_1(q)

        # Dividir C'(x) por a
        C = C_prima * (RealCoefficient(1) / a)
        return C, r_prima
    
    def horner2(self, _a: float, _b: float) -> Tuple[Polynomial, float, float]:
        """
        Descripcion:
            Computa horner cuadratico entre self, y el polinomio x² + ax + b
        """
        # Inicializacion
        a, b = RealCoefficient(-_a), RealCoefficient(-_b)
        C: List[Coefficient] = [RealCoefficient(0) for i in range(self.grade+1)]
        A = self.coeffs
        n = self.grade

        # b_n = an, b_n-1 = a_n-1 + b_n * -a 
        C[n] = A[n]; print(f'C[n]{C[n]} = A[n] {A[n]}');
        C[n-1] = A[n-1] + C[n] * a; print(f'C[n-1]{C[n-1]} = A[n-1] {A[n-1]} + C[n] {C[n]} * a {a}')
        # b_i = a_i + b_i+1 * -a + b_i+2 * -b
        for i in range(n - 2, 0, -1):
            C[i] = A[i] + C[i+1] * a + C[i+2] * b
        # b_0 = a_0 + b_2 * -b
        C[0] = A[0] + C[2] * b
        
        return Polynomial(C[2:]), C[0].value, C[1].value
    def div3(self, Q: Polynomial) -> Tuple[Polynomial, Polynomial]:
        """
        Descripcion:
            Computa la division entre el Polynomial self con el Polynomial Q
            Donde Q es de la forma x² + ax + b
        """
        if Q.grade != 2 and not Q.es_monico():
            raise ValueError(f'q: {Q} debe ser de la forma x² + ax + b')

        b, a = Q.coeffs[0], Q.coeffs[1]
        C, r0, r1 = self.horner2(a.value, b.value)
        return C, Polynomial([RealCoefficient(r0), RealCoefficient(r1)])

    def iter_nw(self, x_n: float) -> float:
        """
        Descripcion:
            Computa una iteracion de newton raphson para Polynomial
            x_n+1 = x_n - P(x_n) / P'(x_n)
            P(x_n), C(x) = P.Horner1(x_n)
            P'(x_n) = C.Horner1(x_n)
        """
        C, P_n = self.horner1(x_n)
        _, P_prima_n = C.horner1(x_n)
        return x_n - P_n / P_prima_n
    def newton(self, x0: float, max_iter=1e3, tol=1e-7) -> Tuple[float, bool]:
        i = 1
        x_ant = x0
        x_curr = self.iter_nw(x_ant)
        while i < max_iter and abs(x_curr - x_ant) > tol:
            x_ant = x_curr
            x_curr = self.iter_nw(x_ant)
            i += 1
        print(i)
        if abs(x_curr - x_ant) > tol:
            return 0, False
        return x_curr, True

    def __call__(self, x: float) -> float:
        # Metodo calleable: evaluacion de Polynomial
        """     
        Descripcion:
            p(x) = a0 + a1x + a2x² + .... + anx^n
            p(x) = a0 + x ( a1 + x ( a2 + x (... ai + x (... an) ) ) )
        acum: "lo acumulado dentro del parentesis que se tiene que mul con x"
        """
        x_0 = RealCoefficient(x)
        acum = RealCoefficient(0)
        for a in self.coeffs[::-1]:
            acum = a + x_0 * acum
        return acum.value
    def __repr__(self) -> str:
        return f'{self.__class__.__name__} ( {self.coeffs} )'
    def __str__(self) -> str:
        return f'Polinomio: {self.coeffs}'
    
    def es_monico(self) -> bool:
        """Determina si el Polynomial self es monico"""
        a_n = self.coeffs[self.grade]
        return a_n.value < 1e-7

class Vizualizer:
    pass
if __name__ == '__main__':
    # Datos
    a0 = RealCoefficient(5)
    a1 = RationalCoefficient(1)
    a2 = IntegerCoefficient(3)
    a3 = RationalCoefficient(0)
    a4 = IntegerCoefficient(12)
    a5 = RealCoefficient(-2)
    p = Polynomial([a0, a1, a2, a3, a4, a5])

    b0 = RealCoefficient(-0.6)
    b1 = RationalCoefficient(1)
    q = Polynomial([b0, b1])
 
    # Variables
    x = 2

    # Debugeo
    print('='*50)
    print(p)
    print(q)
    print(f'raiz de p: {p.newton(5)}')
    

    



    