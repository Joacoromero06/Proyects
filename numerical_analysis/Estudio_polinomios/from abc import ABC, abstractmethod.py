from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Type
from fractions import Fraction
from sympy import symbols, sympify, Poly
from sympy import Integer as SympyInteger, Rational as SympyRational, Float as SympyFloat
import warnings


# ============================================================================
# COEFFICIENT HIERARCHY
# ============================================================================

class Coefficient(ABC):
    """Clase base abstracta para todos los tipos de coeficientes."""
    
    def __init__(self, value):
        self._value = value
    
    @property
    def value(self):
        return self._value
    
    @abstractmethod
    def get_type(self) -> str:
        """Retorna el tipo de coeficiente: 'Integer', 'Rational', 'Real'"""
        pass
    
    @abstractmethod
    def to_integer(self) -> 'IntegerCoefficient':
        """Conversión explícita a Integer"""
        pass
    
    @abstractmethod
    def to_rational(self) -> 'RationalCoefficient':
        """Conversión explícita a Rational"""
        pass
    
    @abstractmethod
    def to_real(self) -> 'RealCoefficient':
        """Conversión explícita a Real"""
        pass
    
    def promote_to(self, target_type: Type['Coefficient']) -> 'Coefficient':
        """Convierte este coeficiente al tipo objetivo"""
        if target_type == IntegerCoefficient:
            return self.to_integer()
        elif target_type == RationalCoefficient:
            return self.to_rational()
        elif target_type == RealCoefficient:
            return self.to_real()
        raise ValueError(f"Tipo desconocido: {target_type}")
    
    @staticmethod
    def _get_dominant_type(coef1: 'Coefficient', coef2: 'Coefficient') -> Type['Coefficient']:
        """Determina el tipo dominante en la jerarquía Integer → Rational → Real"""
        type_hierarchy = {
            IntegerCoefficient: 0,
            RationalCoefficient: 1,
            RealCoefficient: 2
        }
        type1 = type(coef1)
        type2 = type(coef2)
        return type1 if type_hierarchy[type1] > type_hierarchy[type2] else type2
    
    def __add__(self, other: 'Coefficient') -> 'Coefficient':
        """Suma con promoción automática de tipos"""
        dominant_type = self._get_dominant_type(self, other)
        a = self.promote_to(dominant_type)
        b = other.promote_to(dominant_type)
        # AQUÍ: Implementar la lógica de suma según el tipo
        # return dominant_type(a.value + b.value)
        pass
    
    def __sub__(self, other: 'Coefficient') -> 'Coefficient':
        """Resta con promoción automática de tipos"""
        # AQUÍ: Similar a __add__
        pass
    
    def __mul__(self, other: 'Coefficient') -> 'Coefficient':
        """Multiplicación con promoción automática de tipos"""
        # AQUÍ: Similar a __add__
        pass
    
    def __truediv__(self, other: 'Coefficient') -> 'Coefficient':
        """División con promoción automática de tipos"""
        # AQUÍ: Similar a __add__, pero considerar división por cero
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    def __eq__(self, other):
        if not isinstance(other, Coefficient):
            return False
        return self.value == other.value and type(self) == type(other)


class IntegerCoefficient(Coefficient):
    """Coeficiente entero"""
    
    def __init__(self, value: int):
        super().__init__(int(value))
    
    def get_type(self) -> str:
        return "Integer"
    
    def to_integer(self) -> 'IntegerCoefficient':
        return self
    
    def to_rational(self) -> 'RationalCoefficient':
        return RationalCoefficient(self.value, 1)
    
    def to_real(self) -> 'RealCoefficient':
        return RealCoefficient(float(self.value))


class RationalCoefficient(Coefficient):
    """Coeficiente racional (fracción)"""
    
    def __init__(self, numerator: int, denominator: int = 1):
        self._fraction = Fraction(numerator, denominator)
        super().__init__(self._fraction)
    
    def get_type(self) -> str:
        return "Rational"
    
    @property
    def numerator(self):
        return self._fraction.numerator
    
    @property
    def denominator(self):
        return self._fraction.denominator
    
    def to_integer(self) -> 'IntegerCoefficient':
        """Conversión con pérdida de precisión"""
        return IntegerCoefficient(int(self.value))
    
    def to_rational(self) -> 'RationalCoefficient':
        return self
    
    def to_real(self) -> 'RealCoefficient':
        return RealCoefficient(float(self.value))
    
    def __repr__(self):
        return f"RationalCoefficient({self.numerator}/{self.denominator})"


class RealCoefficient(Coefficient):
    """Coeficiente real (punto flotante)"""
    
    def __init__(self, value: float):
        super().__init__(float(value))
    
    def get_type(self) -> str:
        return "Real"
    
    def to_integer(self) -> 'IntegerCoefficient':
        """Conversión con pérdida de precisión"""
        return IntegerCoefficient(int(self.value))
    
    def to_rational(self) -> 'RationalCoefficient':
        """Conversión aproximada a fracción"""
        frac = Fraction(self.value).limit_denominator()
        return RationalCoefficient(frac.numerator, frac.denominator)
    
    def to_real(self) -> 'RealCoefficient':
        return self


# ============================================================================
# COEFFICIENTS CONTAINER
# ============================================================================

class Coefficients:
    """
    Contenedor para una lista de coeficientes.
    Maneja la unificación de tipos y proporciona acceso tipo lista.
    """
    
    def __init__(self, coeffs: List[Coefficient]):
        self._coeffs = coeffs.copy()
        self._unified = False
    
    def unify_type(self) -> None:
        """
        Modifica in-place todos los coeficientes al tipo más dominante.
        Jerarquía: Integer → Rational → Real
        """
        if self._unified or not self._coeffs:
            return
        
        dominant_type = self.get_dominant_type()
        self._coeffs = [c.promote_to(dominant_type) for c in self._coeffs]
        self._unified = True
    
    def get_dominant_type(self) -> Type[Coefficient]:
        """Determina el tipo más complejo presente en la lista"""
        type_hierarchy = {
            IntegerCoefficient: 0,
            RationalCoefficient: 1,
            RealCoefficient: 2
        }
        
        max_level = 0
        dominant = IntegerCoefficient
        
        for coef in self._coeffs:
            level = type_hierarchy[type(coef)]
            if level > max_level:
                max_level = level
                dominant = type(coef)
        
        return dominant
    
    def __getitem__(self, index: int) -> Coefficient:
        """Acceso por índice"""
        return self._coeffs[index]
    
    def __len__(self) -> int:
        return len(self._coeffs)
    
    def __iter__(self):
        return iter(self._coeffs)
    
    def slice(self, start: int, end: int) -> 'Coefficients':
        """Retorna un nuevo Coefficients con un slice de los coeficientes"""
        return Coefficients(self._coeffs[start:end])
    
    def reverse(self) -> 'Coefficients':
        """Retorna un nuevo Coefficients con coeficientes en orden inverso"""
        return Coefficients(list(reversed(self._coeffs)))
    
    def to_list(self) -> List[Coefficient]:
        """Retorna una copia de la lista interna"""
        return self._coeffs.copy()
    
    def __repr__(self):
        return f"Coefficients({self._coeffs})"


# ============================================================================
# POLYNOMIAL
# ============================================================================

class Polynomial:   
    """
    Clase central que representa un polinomio.
    Composición: contiene Coefficients que contiene List[Coefficient]
    """
    
    def __init__(self, coefficients: Coefficients, expression: str = ""):
        """
        Args:
            coefficients: Objeto Coefficients (ya unificado)
            expression: Expresión string original del polinomio
        """
        self._coefficients = coefficients
        self._expression = expression
        self._degree = len(coefficients) - 1
    
    @property
    def coefficients(self) -> Coefficients:
        """Acceso a los coeficientes"""
        return self._coefficients
    
    @property
    def degree(self) -> int:
        """Grado del polinomio"""
        return self._degree
    
    @property
    def expression(self) -> str:
        """Expresión string original"""
        return self._expression
    
    def get_coefficient_type(self) -> Type[Coefficient]:
        """Retorna el tipo de los coeficientes (todos del mismo tipo tras unificación)"""
        if len(self._coefficients) > 0:
            return type(self._coefficients[0])
        return IntegerCoefficient
    
    def __getitem__(self, index: int) -> Coefficient:
        """Acceso directo a coeficientes por índice"""
        return self._coefficients[index]
    
    def evaluate(self, x: float) -> float:
        """Evalúa el polinomio en un punto x"""
        # AQUÍ: Implementar evaluación usando Horner o directa
        pass
    
    def derivative(self) -> 'Polynomial':
        """Retorna la derivada del polinomio como un nuevo Polynomial"""
        # AQUÍ: Implementar cálculo de derivada
        # new_coeffs = ...
        # return Polynomial(Coefficients(new_coeffs))
        pass
    
    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        """Suma de polinomios"""
        # AQUÍ: Implementar suma término a término
        pass
    
    def __sub__(self, other: 'Polynomial') -> 'Polynomial':
        """Resta de polinomios"""
        # AQUÍ: Implementar resta término a término
        pass
    
    def divmod(self, divisor: 'Polynomial') -> Tuple['Polynomial', 'Polynomial']:
        """
        División de polinomios.
        Retorna: (cociente, resto)
        """
        # AQUÍ: Implementar división sintética o algoritmo de división
        pass
    
    def __repr__(self):
        return f"Polynomial(degree={self.degree}, type={self.get_coefficient_type().__name__})"
    
    def __str__(self):
        if self._expression:
            return self._expression
        # AQUÍ: Construir representación string desde coeficientes
        return f"Polynomial of degree {self.degree}"


# ============================================================================
# HORNER EVALUATOR
# ============================================================================

class Horner:
    """
    Clase independiente para evaluación y división usando el método de Horner.
    Conoce y trabaja con Polynomial y Coefficient.
    """
    
    @staticmethod
    def simple(poly: Polynomial, root: Coefficient) -> Tuple[Polynomial, Coefficient]:
        """
        Método de Horner simple: divide poly por (x - root)
        
        Args:
            poly: Polinomio a dividir
            root: Raíz (valor de r en x - r)
        
        Returns:
            (cociente_polynomial, resto_coefficient)
        """
        # AQUÍ: Implementar algoritmo de Horner simple
        # 1. Extraer coeficientes del polinomio
        # 2. Aplicar esquema de Horner
        # 3. Crear nuevo Polynomial para cociente
        # 4. Retornar resto como Coefficient
        pass
    
    @staticmethod
    def double(poly: Polynomial, a: Coefficient, b: Coefficient) -> Tuple[Polynomial, Polynomial]:
        """
        Método de Horner doble: divide poly por (x-a)(x-b) = x² - (a+b)x + ab
        
        Args:
            poly: Polinomio a dividir
            a, b: Raíces del divisor cuadrático
        
        Returns:
            (cociente_polynomial, resto_polynomial)
            El resto es un polinomio de grado ≤ 1
        """
        # AQUÍ: Implementar algoritmo de Horner doble
        # 1. Calcular coeficientes del divisor cuadrático
        # 2. Aplicar esquema de Horner doble
        # 3. Crear Polynomial para cociente
        # 4. Crear Polynomial para resto (grado ≤ 1)
        pass


# ============================================================================
# POLYNOMIAL BUILDER
# ============================================================================

class PolynomialBuilder:
    """
    Factory para crear polinomios de diferentes formas.
    Centraliza la lógica de construcción y parsing.
    """
    
    @staticmethod
    def from_string(expr: str) -> Polynomial:
        """
        Construye un Polynomial desde una expresión string.
        Usa SymPy para parsing y determina tipos de coeficientes.
        
        Args:
            expr: Expresión matemática como "2*x**2 + 3*x + 1"
        
        Returns:
            Polynomial con coeficientes unificados
        """
        x = symbols('x')
        
        # Parsear con SymPy
        sympy_expr = sympify(expr)
        sympy_poly = Poly(sympy_expr, x)
        sympy_coeffs = sympy_poly.all_coeffs()
        
        # Convertir coeficientes de SymPy a nuestros Coefficient
        coefficients = []
        for sc in sympy_coeffs:
            coef = PolynomialBuilder._sympy_to_coefficient(sc)
            coefficients.append(coef)
        
        # Crear Coefficients y unificar tipos
        coeff_list = Coefficients(coefficients)
        coeff_list.unify_type()
        
        # Crear y retornar Polynomial
        return Polynomial(coeff_list, expression=expr)
    
    @staticmethod
    def _sympy_to_coefficient(sympy_coef) -> Coefficient:
        """Convierte un coeficiente de SymPy a nuestro Coefficient"""
        if isinstance(sympy_coef, SympyInteger):
            return IntegerCoefficient(int(sympy_coef))
        elif isinstance(sympy_coef, SympyRational):
            return RationalCoefficient(int(sympy_coef.p), int(sympy_coef.q))
        elif isinstance(sympy_coef, SympyFloat):
            return RealCoefficient(float(sympy_coef))
        else:
            # Expresión simbólica: convertir a float
            warnings.warn(f"Expresión simbólica {sympy_coef} convertida a float")
            return RealCoefficient(float(sympy_coef.evalf()))
    
    @staticmethod
    def from_coefficients(coeffs: List[Union[int, float, Fraction, Coefficient]]) -> Polynomial:
        """
        Construye un Polynomial desde una lista de valores.
        
        Args:
            coeffs: Lista de coeficientes (int, float, Fraction o Coefficient)
                   Orden: [a_n, a_{n-1}, ..., a_1, a_0] (grado mayor a menor)
        
        Returns:
            Polynomial con coeficientes unificados
        """
        coefficient_objects = []
        
        for c in coeffs:
            if isinstance(c, Coefficient):
                coefficient_objects.append(c)
            elif isinstance(c, int):
                coefficient_objects.append(IntegerCoefficient(c))
            elif isinstance(c, Fraction):
                coefficient_objects.append(RationalCoefficient(c.numerator, c.denominator))
            elif isinstance(c, float):
                coefficient_objects.append(RealCoefficient(c))
            else:
                raise TypeError(f"Tipo no soportado: {type(c)}")
        
        coeff_list = Coefficients(coefficient_objects)
        coeff_list.unify_type()
        
        return Polynomial(coeff_list)
    
    @staticmethod
    def from_roots(roots: List[float]) -> Polynomial:
        """
        Construye un Polynomial a partir de sus raíces.
        P(x) = (x - r1)(x - r2)...(x - rn)
        
        Args:
            roots: Lista de raíces
        
        Returns:
            Polynomial expandido
        """
        # AQUÍ: Implementar construcción desde raíces
        # Multiplicar factores (x - r_i) sucesivamente
        pass


# ============================================================================
# ROOT FINDING STRATEGIES
# ============================================================================

class RootFindingStrategy(ABC):
    """Estrategia base para encontrar raíces"""
    
    @abstractmethod
    def find_roots(self, poly: Polynomial) -> List[Coefficient]:
        """
        Encuentra raíces del polinomio.
        
        Args:
            poly: Polinomio a analizar
        
        Returns:
            Lista de raíces como Coefficients
        """
        pass
    
    @abstractmethod
    def can_handle(self, poly: Polynomial) -> bool:
        """Verifica si esta estrategia puede manejar el polinomio dado"""
        pass


class RationalRootFinder(RootFindingStrategy):
    """Estrategia para encontrar raíces racionales (Teorema de raíces racionales)"""
    
    def can_handle(self, poly: Polynomial) -> bool:
        """Solo funciona con coeficientes Integer o Rational"""
        coef_type = poly.get_coefficient_type()
        return coef_type in [IntegerCoefficient, RationalCoefficient]
    
    def find_roots(self, poly: Polynomial) -> List[Coefficient]:
        """
        Aplica el teorema de raíces racionales.
        Prueba todos los candidatos p/q donde p|a_0 y q|a_n
        """
        if not self.can_handle(poly):
            raise ValueError("RationalRootFinder solo trabaja con coeficientes Integer/Rational")
        
        # AQUÍ: Implementar algoritmo de búsqueda de raíces racionales
        # 1. Obtener divisores del término independiente y coeficiente principal
        # 2. Generar candidatos p/q
        # 3. Evaluar polinomio en cada candidato
        # 4. Retornar raíces encontradas
        pass


class BairstowSolver(RootFindingStrategy):
    """Estrategia para encontrar raíces usando el método de Bairstow"""
    
    def can_handle(self, poly: Polynomial) -> bool:
        """Funciona con cualquier tipo, pero convierte a Real si es necesario"""
        return True
    
    def find_roots(self, poly: Polynomial) -> List[Coefficient]:
        """
        Aplica el método iterativo de Bairstow para encontrar factores cuadráticos.
        """
        # Si el polinomio no tiene coeficientes Real, convertir
        if poly.get_coefficient_type() != RealCoefficient:
            poly = self._convert_to_real(poly)
        
        # AQUÍ: Implementar algoritmo de Bairstow
        # 1. Iterar para encontrar factores cuadráticos x² + px + q
        # 2. Deflactar el polinomio usando Horner.double()
        # 3. Resolver cada factor cuadrático
        # 4. Retornar todas las raíces
        pass
    
    def _convert_to_real(self, poly: Polynomial) -> Polynomial:
        """Convierte un polinomio a coeficientes Real"""
        real_coeffs = [c.to_real() for c in poly.coefficients]
        return Polynomial(Coefficients(real_coeffs), poly.expression)


# ============================================================================
# ROOT BOUNDING
# ============================================================================

class RootBounding:
    """Métodos para calcular cotas de raíces de polinomios"""
    
    @staticmethod
    def cauchy_bound(poly: Polynomial) -> float:
        """
        Calcula la cota de Cauchy para las raíces.
        Todas las raíces están dentro del círculo |z| < R
        """
        # AQUÍ: Implementar fórmula de la cota de Cauchy
        pass
    
    @staticmethod
    def lagrange_bound(poly: Polynomial) -> float:
        """
        Calcula la cota de Lagrange para las raíces.
        """
        # AQUÍ: Implementar fórmula de la cota de Lagrange
        pass
    
    @staticmethod
    def kojima_bound(poly: Polynomial) -> float:
        """
        Calcula la cota de Kojima (refinamiento de Cauchy)
        """
        # AQUÍ: Implementar fórmula de la cota de Kojima
        pass


# ============================================================================
# VISUALIZER
# ============================================================================

class PolynomialVisualizer:
    """Visualización de polinomios y sus raíces"""
    
    @staticmethod
    def plot(poly: Polynomial, x_range: Tuple[float, float] = (-10, 10)):
        """
        Grafica el polinomio en el rango especificado
        
        Args:
            poly: Polinomio a graficar
            x_range: Rango (x_min, x_max)
        """
        # AQUÍ: Implementar graficación con matplotlib
        pass
    
    @staticmethod
    def plot_with_roots(poly: Polynomial, roots: List[Coefficient], 
                       x_range: Tuple[float, float] = (-10, 10)):
        """
        Grafica el polinomio y marca sus raíces
        
        Args:
            poly: Polinomio a graficar
            roots: Lista de raíces para marcar
            x_range: Rango (x_min, x_max)
        """
        # AQUÍ: Implementar graficación con matplotlib
        # Marcar raíces con puntos especiales
        pass


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo 1: Crear polinomio desde string
    print("=== Ejemplo 1: Desde string ===")
    poly1 = PolynomialBuilder.from_string("2*x**2 + 3*x + 1")
    print(poly1)
    print(f"Grado: {poly1.degree}")
    print(f"Tipo de coeficientes: {poly1.get_coefficient_type().__name__}")
    print(f"Coeficientes: {[c for c in poly1.coefficients]}")
    
    # Ejemplo 2: Crear desde lista de coeficientes
    print("\n=== Ejemplo 2: Desde lista ===")
    poly2 = PolynomialBuilder.from_coefficients([1, 0, -4])  # x² - 4
    print(poly2)
    print(f"Coeficientes: {[c for c in poly2.coefficients]}")
    
    # Ejemplo 3: Coeficientes mixtos con unificación
    print("\n=== Ejemplo 3: Unificación de tipos ===")
    coeffs = Coefficients([
        IntegerCoefficient(2),
        RationalCoefficient(1, 3),
        RealCoefficient(1.5)
    ])
    print(f"Antes de unificar: {coeffs}")
    print(f"Tipo dominante: {coeffs.get_dominant_type().__name__}")
    coeffs.unify_type()
    print(f"Después de unificar: {coeffs}")
    
    print("\n=== Sistema listo para implementar algoritmos ===")