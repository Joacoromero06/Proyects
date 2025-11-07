from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from fractions import Fraction


class Coefficient(ABC):
    """ Plantilla abstracta para cualquier coeficiente """
    
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value
    
    @abstractmethod
    def get_type(self)->str:
        """ 
        Retorna el tipo del coeficiente 'self'.
        ej: 'Real', 'Integer', 'Rational'
        """
        pass

    @abstractmethod
    def to_integer(self)->IntegerCoefficient:
        """ 
        Retorna el coeficiente actual en formato IntegerCoefficient
        Conversion to Integer
        """
        pass

    @abstractmethod 
    def to_rational(self)->RationalCoefficient:
        """
        REtorna el coeficiente actual en formato RAtionalCoefficient
        Conversion a Rational
        """
        pass

    @abstractmethod
    def to_real(self)->RealCoefficient:
        """
        Retorna el coeficiente actual en formato REalCoefficient
        Conversion a Float
        """

    def promote_to(self, target_type: str)->Coefficient:
        """ Convierte self, al target_type """
        if target_type == 'Integer':
            return self.to_integer()
        if target_type == 'Rational':
            return self.to_rational()
        if target_type == 'Real':
            return self.to_real()
        raise ValueError(f'Intento de promote_to() con tipo desconocido: {target_type}')
    
    @staticmethod
    def _get_dominant_type(coef1: Coefficient, coef2: Coefficient)->str:
        """
        Descripcion:
            Metodo utilitaria de la clase
            No es de instancia
        Objetivo:
            Computa el tipo dominante entre dos coeficientes
        """
        map_h = {# hashmap(tipo)-> herarchy
            'Integer': 1,
            'Rational': 2,
            'Real': 3
        }
        type1 = coef1.get_type()
        type2 = coef2.get_type()
        max_type = type1 if map_h[type1] > map_h[type2] else type2
        return max_type
    
    def __add__(self, other: Coefficient)-> Coefficient:
        """Iguala los tipos y computa la suma"""
        dominant_type = Coefficient._get_dominant_type(self, other)
        a = self.promote_to(dominant_type)
        b = other.promote_to(dominant_type)

        if dominant_type == 'Integer':
            return IntegerCoefficient(a.value + b.value)
        if dominant_type == 'Rational':
            return RationalCoefficient.from_fraction(a.value + b.value)
        if dominant_type == 'Real':
            return RealCoefficient(a.value + b.value)  
        return IntegerCoefficient(0)
    def __sub__(self, other: Coefficient)-> Coefficient:
        """Iguala los tipos y computa la suma"""
        dominant_type = Coefficient._get_dominant_type(self, other)
        a = self.promote_to(dominant_type)
        b = other.promote_to(dominant_type)

        if dominant_type == 'Integer':
            return IntegerCoefficient(a.value - b.value)
        if dominant_type == 'Rational':
            return RationalCoefficient.from_fraction(a.value - b.value)
        if dominant_type == 'Real':
            return RealCoefficient(a.value - b.value)    
        return IntegerCoefficient(0)
    def __mul__(self, other: Coefficient)-> Coefficient:
        """Iguala los tipos y computa la suma"""
        dominant_type = Coefficient._get_dominant_type(self, other)
        a = self.promote_to(dominant_type)
        b = other.promote_to(dominant_type)

        if dominant_type == 'Integer':
            return IntegerCoefficient(a.value * b.value)
        if dominant_type == 'Rational':
            return RationalCoefficient.from_fraction(a.value * b.value)
        if dominant_type == 'Real':
            return RealCoefficient(a.value * b.value)
        return IntegerCoefficient(0)
    def __truediv__(self, other: Coefficient)-> Coefficient:
        """Iguala los tipos y computa la suma"""
        dominant_type = Coefficient._get_dominant_type(self, other)
        a = self.promote_to(dominant_type)
        b = other.promote_to(dominant_type)

        if dominant_type == 'Integer':
            return IntegerCoefficient(a.value / b.value)
        if dominant_type == 'Rational':
            return RationalCoefficient.from_fraction(a.value / b.value)
        if dominant_type == 'Real':
            return RealCoefficient(a.value / b.value)
        return IntegerCoefficient(0)
        
    
    def __repr__(self):
        return f'{self.__class__.__name__}( {self.value} )'
    def __eq__(self, other: Coefficient):
        if not isinstance(other, Coefficient):
            return False
        return self.value == other.value and type(self) == type(other)
    def __str__(self):
        return f'{self.get_type()}: ( {self.value} ) '

class IntegerCoefficient(Coefficient):
    def __init__(self, value):
        if not isinstance(value, int):
            raise TypeError(f'value: {value} no es type int')
        super().__init__(value)
    
    def get_type(self):
        return 'Integer'
    
    def to_integer(self):
        return self
    
    def to_rational(self):
        return RationalCoefficient(self.value, 1)
    
    def to_real(self):
        return RealCoefficient( float(self.value) )
    
class RationalCoefficient(Coefficient):
    def __init__(self, numerator: int, denominator: int = 1):
        self._fraction = Fraction(numerator, denominator)
        super().__init__(self._fraction)
    
    @classmethod
    def from_fraction(cls, fraction: Fraction):
        return cls(fraction.numerator, fraction.denominator)
    
    def get_type(self):
        return 'Rational'
    
    @property
    def numerator(self):
        return self._fraction.numerator
    
    @property
    def denominator(self):
        return self._fraction.denominator
    
    def to_integer(self):
        """
        Problema: 
            Dos atributos almacenan la misma informacion
            self._value = self._fraction
        Descripcion:
            El objeto Fraction ya tiene implementado el casteo int()
        """
        return IntegerCoefficient( int(self.value) )
    
    def to_rational(self):
        return self
    
    def to_real(self):
        """
        Problema: 
            Dos atributos almacenan la misma informacion
            self._value = self._fraction
        Descripcion:
            El objeto Fraction ya tiene implementado el casteo float()
        """
        return RealCoefficient( float(self.value) )
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.numerator} / {self.denominator})'

class RealCoefficient(Coefficient):
    def __init__(self, value: float):
        super().__init__( float(value) )
    
    def get_type(self):
        return 'Real'
    
    def to_integer(self):
        return IntegerCoefficient( int(self.value) )
    
    def to_rational(self):
        fraction = Fraction(self.value).limit_denominator(100)
        return RationalCoefficient(fraction.numerator, fraction.denominator)
    
    def to_real(self):
        return self



if __name__ == '__main__':
    a = RealCoefficient(3.14)
    b = RationalCoefficient(6, 16)
    c = IntegerCoefficient(10)

    l = []
    data = [a,b,c]
    i = 0
    for c in data:
        l.append(c.to_integer())
        l.append(c.to_rational())
        l.append(c.to_real())

    def suma(a,b):
        print(f'Sumando {a} con {b}.\nLa suma es: {a + b}')
    
    print('='*50)
    i = 0
    while i < len(l) -1 :
        l_i = l[i]
        l_sig = l[i+1]
        suma( l_i, l_sig)
        i += 1
    print('='*50)
    
    


