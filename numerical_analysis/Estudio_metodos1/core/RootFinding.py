from dataclasses import dataclass 
from typing import List, Optional, Any
from FunctionHandler import FunctionHandler
from collections import deque
import pandas as pd
from datetime import datetime

@dataclass(frozen=True)
class IterationResult:
    """Resultado de una unica iteracion"""
    iteration: int
    x_n: float
    f_x_n: float
    error_x: Optional[float] = None
    error_f: Optional[float] = None

@dataclass(frozen=True)
class ExecutionResult:
    """Resultado total de una ejecucion"""
    method_name: str
    converged: bool
    root: Optional[float]
    iterations: List[IterationResult]
    final_error_x: Optional[float] = None
    final_error_f: Optional[float] = None
    message: str = ""

    @property
    def total_iteration(self):
        """Calcula la cantidad de iteraciones"""
        return len(self.iterations)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convierte los resultados a un DataFrame de pandas"""
        if not self.iterations:
            return pd.DataFrame(columns=['n', 'x_n', 'f(x_n)', 'error_x', 'error_f'])
        
        data = {
            'n': [it.iteration for it in self.iterations],
            'x_n': [it.x_n for it in self.iterations],
            'f(x_n)': [it.f_x_n for it in self.iterations],
            'error_x': [it.error_x for it in self.iterations],
            'error_f': [it.error_f for it in self.iterations]

        }
        return pd.DataFrame(data).set_index('n')
    
    def to_excel(self, filename: Optional[str] = None) -> str:
        """
        Descripcion:
            Exporta los resultados a un archivo Excel.
            filename: Nombre del archivo (opcional). Si no se provee,
            genera automáticamente: "{method_name}_results.xlsx"
        Retorna:
            Ruta del archivo creado.
        """
        
        
        # Generar nombre automático si no se proporciona
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.method_name}_{timestamp}.xlsx"
        
        # Asegurar extensión .xlsx
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        
        # Crear DataFrame
        df = self.to_dataframe()
        
        # Crear Excel con múltiples hojas
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Hoja 1: Iteraciones
            df.to_excel(writer, sheet_name='Iteraciones')
            
            # Hoja 2: Resumen
            summary_data = {
                'Método': [self.method_name],
                'Convergió': [self.converged],
                'Raíz encontrada': [self.root if self.root else 'N/A'],
                'Iteraciones totales': [self.total_iteration],
                'Error final (x)': [self.final_error_x if self.final_error_x else 'N/A'],
                'Error final (f)': [self.final_error_f if self.final_error_f else 'N/A'],
                'Mensaje': [self.message]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
        
        return filename
    

from abc import ABC, abstractmethod

class RootFinder(ABC):
    """
    Descrpcion:
        Clase abstracta para metodos de busqueda de raices usando el patron
    
    1. Contiene la configuracion general que comparten los metodos
    2. Implemente la estructura del algoritmo de busqueda de raices
    3. Maneja las distintas formas de convergencia
    4. Maneja los errores que pueden ocurrir
    """ 

    def __init__(self, function: FunctionHandler, tol: float = 1e-7, 
                 max_iter: int = 100, check_error_x: bool = True, check_error_f:bool = True):
        """
        function:       objeto FunctionHandler de f(x)
        tol:            Tolerancia para determinar la convergencia
        max_iter:       Numero maximo de iteraciones
        check_error_x:  Verifica si |x_n - x_n-1| < tol
        check_error_f:  Verifica si |f(x_n)| < tol
        """
        
        self.function = function
        self.tol = tol
        self.max_iter = max_iter
        self.check_error_x = check_error_x
        self.check_error_f = check_error_f

    def find_root(self, *args: Any, **kwargs: Any) -> ExecutionResult:
        """
        Algoritmo general de busqueda de raices
        Pasos:
        1. Validar entradas especificas para el metodo
        2. Iniciar historia de aproximaciones para generar la proxima aproximacion
        3. Iterar hasta converger o max_iter
        4. Retornar resultado inmutable
        """
        # Inicializacion de lista de iteraciones
        iteration_data: List[IterationResult] = []

        # Valida las aproximaciones iniciales
        try:
            self._validate_inputs(*args, **kwargs)
        except ValueError as e:
            return ExecutionResult(
                method_name=self.__class__.__name__, 
                converged=False,
                root= None, 
                iterations=[], 
                message=f'Error de validacion: {str(e)}'
                )
        
        # Inicializacion de aproximaciones iniciales
        try:
            history = self._initializate(*args, **kwargs)
        except ValueError as e:
            return ExecutionResult(
                method_name=self.__class__.__name__, 
                converged=False,
                root= None, 
                iterations=[], 
                message=f'Error de inicializacion: {str(e)}'
                )
        
        # Algoritmo general
        for i in range(self.max_iter):            
            x_curr = history[-1] # la ultima aproximacion es la actual

            # Evaluar la funcion en la aproximacion
            try:
                f_x_curr = self.function(x_curr)
            except (ZeroDivisionError, ValueError, OverflowError) as e :
                return ExecutionResult(
                    method_name=self.__class__.__name__,
                    converged=False,
                    root=x_curr,
                    iterations=iteration_data,
                    message=f"Error numerico en iteracion {i}: {str(e)} "
                    )    
            
            # Calcular errores
            #print(f'history es: {history} iteracion nro: {i}')
            error_x = abs(x_curr - history[-2]) if len(history) >= 2 else None
            error_f = abs(f_x_curr)

            # Guardar rtdos de la iteracion
            iteration_data.append(IterationResult(i, x_curr, f_x_curr, error_x, error_f))

            # Verificar convergencia
            converged = self._check_convergence(error_x, error_f)
            if converged:
                return ExecutionResult(
                    method_name=self.__class__.__name__,
                    converged=True,
                    root=x_curr,
                    iterations=iteration_data,
                    final_error_x=error_x,
                    final_error_f=error_f,
                    message="Convergencia alcanzada."
                )
            
            # Calcula la siguiente aproximacion
            try: 
                print(f'history: {history}, iteracion = {i}')
                x_next = self._iterate(history)
                print(f'history luego de obtener x_next: {history}')
            except (ValueError, ZeroDivisionError, OverflowError) as e:
                return ExecutionResult(
                    method_name=self.__class__.__name__,
                    converged=False,
                    root=x_curr,
                    iterations=iteration_data,
                    message=f"Error en iteración {i}: {str(e)}"
                )
            history.append(x_next)
            print(f'history luego de agregar x_next: {history}')
            print()

            # Mantener el historial necesario
            max_history = self._get_history_size()
            if len(history) > max_history:
                print('borra?  ', history)
                history.popleft()
        
        
        return ExecutionResult(
            method_name=self.__class__.__name__,
            converged=False,
            root=history[-1],
            iterations=iteration_data,
            final_error_x=error_x if len(history) >= 2 else None,
            final_error_f=error_f,
            message=f"No convergió en {self.max_iter} iteraciones."
        )
    
    def _check_convergence(self, error_x: Optional[float], error_f: Optional[float]):
        """Verifica si se alcanzo la convergencia de todos los criterios"""
        conditions = []
        if self.check_error_x and error_x is not None:
            conditions.append( error_x < self.tol )
        if self.check_error_f and error_f is not None:
            conditions.append( error_f < self.tol )
        
        return all(conditions) if conditions else False
    
    @abstractmethod##mequede en ver los metodos desp de root_finding
    def _validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        """
        Descripcion:
            Valida las entradas del algoritmo, tarea de cada metodo en particular
            Si no son validas lanza ValueError
        """
        pass

    @abstractmethod
    def _initializate(self, *args: Any, **kwargs: Any) -> deque:
        """
        Descripcion:
            Inicializa el historial de aproximaciones
            Cada algoritmo necesita llevar en cuenta distintas cantidades de aproxs
            El ultimo elemento es la ultima aproximacion.
        """
        pass

    @abstractmethod
    def _iterate(self, history: deque) -> float:
        """
        Descripcion:
            Efectua una iteracion del algoritmo
            Obtiene el siguiente termino de la sucesion
        Args:
            history: 
                objeto deque paitonic donde se llevan las aproximaciones 
                que necesita cada algoritmo para generar la prox iteracion
        Returns:
            x_next: 
                punto flotante, la siguiente aproximacion generada por el algoritmo
        """
        pass
    
    @abstractmethod
    def _get_history_size(self) -> int:
        """
        Descripcion:
            Calcula la cantidad de aproximaciones que el metodo concreto
            necesita
            Podria verse como una propiedad de cada metodo, igual lo hice como
            metodo
        """