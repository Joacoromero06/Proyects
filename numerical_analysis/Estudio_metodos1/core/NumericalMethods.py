from RootFinding import RootFinder
from typing import Optional, List, Any
class Newton(RootFinder):
    def _validate_inputs(self, x0: float):
        """Validamos que el input sea un numero"""
        if not isinstance(x0, (int, float)):
            raise ValueError("x0 debe ser un numero paitonic")
    
    def _initializate(self, x0: float) -> tuple[None, float]:
        """Retornamos las aproximaciones iniciales para el metodo newton"""
        return (None, x0)
    
    def _iterate(self, x_curr: float) -> tuple[float, float]:
        """Obtenemos el siguiente termino de la secuencia y retornamos el nuevo termino y el anterior"""
        f_x = self.function(x_curr)
        f_prime_x = self.function.f_prime(x_curr)

        if f_prime_x == 0:
            raise ValueError("Derivada cero, Newton no puede continuar")
        
        x_next = x_curr - f_x / f_prime_x
        return x_curr, x_next