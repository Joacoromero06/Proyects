from dataclasses import dataclass, field
from typing import List, Optional, Any
import pandas as pd
from FunctionHandler import FunctionHandler

@dataclass(frozen=True)
class IterationResult:
    """Resultado de una unica iteracion"""
    iteration: int
    x_n: float
    f_x_n: float
    error: Optional[float] = None

@dataclass(frozen=True)
class ExecutionResult:
    """Resultado total de una ejecucion"""
    method_name: str
    converged: bool
    root: Optional[float]
    iterations: List[IterationResult]
    final_error: Optional[float] = None
    message: str = ""

    @property
    def total_iteration(self):
        """Calcula la cantidad de iteraciones"""
        return len(self.iterations)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convierte los resultados a un DataFrame de pandas"""
        if not self.iterations:
            return pd.DataFrame(columns=['n', 'x_n', 'f(x_n)', 'error'])
        
        data = {
            'n': [it.iteration for it in self.iterations],
            'x_n': [it.x_n for it in self.iterations],
            'f(x_n)': [it.f_x_n for it in self.iterations],
            'error': [it.error for it in self.iterations]
        }
        return pd.DataFrame(data).set_index('n')
    
from abc import ABC, abstractmethod

class RootFinder(ABC):
    """
    Descrpcion:
        Clase abstracta para metodos de busqueda de raices usando el patron
    
    Contiene la configuracion general que comparten los metodos
    Implemente la estructura del algoritmo de busqueda de raices
    """ 

    def __init__(self, function: FunctionHandler, tol: float = 1e-7, max_iter: int = 100):
        self.function = function
        self.tol = tol
        self.max_iter = max_iter

    def find_root(self, *args: Any, **kwargs: Any) -> ExecutionResult:
        iteration_data: List[IterationResult] = []

        try:
            # Valida las aproximaciones iniciales
            self._validate_inputs(*args, **kwargs)

            # Inicializa las variables
            x_prev, x_curr = self._initializate(*args, **kwargs)
        except ValueError as e:
            return ExecutionResult(self.__class__.__name__, False, 
                                   None, [], message=str(e))
        for i in range(self.max_iter):            
            try:

                f_x_curr = self.function(x_curr)
                error = abs(x_curr - x_prev) if x_prev is not None else None

                iteration_data.append(IterationResult(i, x_curr, f_x_curr, error))

                if error is not None and error < self:
                    return ExecutionResult(
                        method_name=self.__class__.__name__,
                        converged=True,
                        root=x_curr,
                        iterations=iteration_data,
                        final_error=error,
                        message="Convergencia alcanzada"
                        )
                
                x_prev, x_curr = self._iterate(x_prev, x_curr)
            except (ZeroDivisionError, ValueError, OverflowError) as e :
                return ExecutionResult(
                    method_name=self.__class__.__name__,
                    converged=False,
                    root=x_curr,
                    iterations=iteration_data,
                    message=f"Error numerico en iteracion {i}: {str(e)} "
                    )
        return ExecutionResult(
            method_name=self.__class__.__name__,
            converged=False,
            root=x_curr,
            iterations=iteration_data,
            message=f"No se alcanzo la convergencia en {self.max_iter} iteraciones"
        )
    
    @abstractmethod
    def _validate_inputs(self, *args: Any, **kwargs: Any):
        """Valida las entradas necesarias para la efectuacion del algoritmo"""
        pass

    @abstractmethod
    def _initializate(self, *args: Any, **kwargs: Any) -> tuple(Optional[float], float):
        """Inicializa los valores para la primera iteracion"""
        pass

    @abstractmethod
    def _iterate(self, x_prev: float, x_curr: float) -> tuple[float, float]:
        """
        Efectua una iteracion del algoritmo
        Obtiene el siguiente termino de la sucesion
        Actualiza el x_prev y el x_curr para la sig iteracion
        """
        pass
    