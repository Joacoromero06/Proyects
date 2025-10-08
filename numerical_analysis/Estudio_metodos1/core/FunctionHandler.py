from typing import Any
import sympy
from sympy import symbols, sympify, diff, latex, lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from functools import cached_property 

class FunctionHandler:
    """
    Encapsula el concepto de funcion matematica
    Lo hacemos con sympy, maneja matematica simbolica

    Proporciona ademas distintas representaciones:
        como str, numpy_fn, formato latex, sus derivadas

    Es inmutable, menos en la parte de un proceso de lazyevaluation para
    optimizar el calculo de evaluaciones de la funcion en un punto 
    y de sus derivadas
    """
    def __init__(self, expr_str: str, var: str = 'x') -> None:
        """
        Inicializa el objeto creando las representaciones de la funcion

        Args:
            expr_str: la funcion en string porque es la forma del input
            var: la variable que puede ser especificada
        
        Raises:
            Value Error: si la expresion no se puede parsear
        """

        self.expr_str = expr_str
        self.var_str = var
        self.var = symbols(self.var_str)

        # Manejo de errores para expresion invalidas
        try:
            transformations = (standard_transformations + 
                   (implicit_multiplication_application,))
            self.expr = parse_expr(self.expr_str, transformations=transformations)
        except:
            raise ValueError(f"La expresion {self.expr_str} no es valida")
        
        self.expr_prime = diff(self.expr)
        self.expr_2prime = diff(self.expr_prime)
    
    @cached_property
    def f(self):
        """
        Descripcion:
            @cached property es un decorador, que permite definir en este caso
            un metodo de la clase que calcula un atributo de la clase, pero
            solo la primera vez. util para mas eficiencia y sentido
        Retorno:
            Funcion numerica evaluable        
        """
        return lambdify(self.var, self.expr, 'numpy')

    @cached_property
    def f_prime(self):
        """ Funcion derivada numerica evaluable """
        return lambdify(self.var, self.expr_prime, 'numpy')
    
    @cached_property
    def f_2prime(self):
        """ Funcion derivada segunda numerica evaluable """
        return lambdify(self.var, self.expr_2prime, 'numpy')
    
    @cached_property
    def latex_str(self):
        """ Retorna la expresion simbolica en formato latex """
        return latex(self.expr)
    
    # metodos clasicos en python poo
    def __call__(self, value):
        """
        Descripcion:
            Permite que el objeto sea llamable, como una funcion
            mi_obj(3) = mi_obj.f(3)
        """
        return self.f(value)

    def __str__(self) -> str:
        return f"f({self.var}) = {self.expr_str}"
    
    def __repr__(self) -> str:
        return f"FunctionHandler('{self.expr_str}', '{self.var_str}')"
    
if __name__ == '__main__':
    print("--- Probando una función válida ---")
    try:
        # Crear
        func = FunctionHandler("cos x - sin x + exp x")
        x0 = 2
        
        # pruebas de metodos python
        print(f'call: f({x0}) = {func(x0)}')
        print(f'str: {func}')
        print(f'repr: {repr(func)}')

        # prueba mis metodos
        print(f'df({x0}) = {func.f_prime(x0)}')
        print(f'ddf({x0}) = {func.f_2prime(x0)}')
        print(f'latex: {func.latex_str}')
        
    except ValueError as e:
        print(e)#printea mi error