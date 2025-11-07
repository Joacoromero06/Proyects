from RootFinding import RootFinder
from FunctionHandler import FunctionHandler
from typing import Any, Union
from collections import deque
class Newton(RootFinder):
    """
    Descripćion:
        Clase Newton, hereda de la plantilla abstracta RootFinder
        Encuentra raices de funciones definidas por el usuario
        Se relaciona con el algoritmo find_root() y la sucesion de Newton
    """
    
    def _validate_inputs(self, x0: float) -> None:
        """Validamos que el input sea un numero"""
        if not isinstance(x0, (int, float)):
            raise ValueError("x0 debe ser un numero paitonic")
    
    def _initializate(self, x0: float) -> deque:
        """Retornamos las aproximaciones iniciales para el metodo newton"""
        return deque([x0])
    
    def _iterate(self, history: deque) -> float:
        """
        Descripcion:
            Obtenemos el siguiente termino de la secuencia 
            Retornamos el nuevo termino que sera agregado a history en find_root()
            Dejamos la aproximacion anterior en history
                razon: error_x se calcula con |x_{n} - x_{n-1}|
        """
        x_curr = history[-1]
        f_x = self.function(x_curr)
        f_prime_x = self.function.f_prime(x_curr)

        if f_prime_x == 0:
            raise ValueError("Derivada cero, Newton no puede continuar")
        
        x_next = x_curr - f_x / f_prime_x
        return x_next

    def _get_history_size(self) -> int:
        """
        Retornamos 2 porque newton necesita solo la iteracion n-sima y
        la anterior para calcular el error |x_{n} - x_{n-1}| """
        return 2

class Biseccion(RootFinder):
    """
    Descripcion:
        Clase Biseccion, hereda de la plantilla abstracta RootFinder.
        Implementa los metodos abstractos.
        Se relaciona con la sucesion de biseccion y el algoritmo find_root()
    """
    def __init__(self, function: FunctionHandler, tol: float = 1e-7, 
                 max_iter: int = 100, check_error_x: bool = False, check_error_f: bool = True):
        super().__init__(function, tol, max_iter, check_error_x, check_error_f)

    def _validate_inputs(self, a: Union[float, int], b: Union[float, int]) -> None:
        """
        Descripcion:
            a tiene que ser menor que b
            f(a) y f(b) tienen que tener signois opuestos
            Biseccion es un metodo cerrado
        """
        
        if a >= b:
            raise ValueError(f'Debe cumplirse que a: {a} es menor que b: {b}')
        f_a: float = self.function(a)
        f_b: float = self.function(b)
        if signo(f_a) == signo(f_b):
            raise ValueError(f'a: {a} y b: {b}. Deben ser tales que tengan signo opuesto')

    def _initializate(self, a: Union[float, int], b: Union[float, int]) -> deque:
        """
        Descripcion:
            Definicion del metodo abstracto inicializa el historial para la ejecucion
            del algoritmo find_root() a partir de las aprox iniciales
        Razon:
            Quizas es extraño que se inicialize con el punto medio.
            Es necesario debido a find_root() calcula x_next con iterate()
        """
        
        c = (a + b) / 2
        return deque([a,b,c])
    
    def _iterate(self, history: deque) -> float:
        """
        Descripcion:
            Calcula la siguiente aproximacion y la retorna.
        history:
            historial de las aproximaciones necesarias para generar el sig term
            de la sucesion de biseccion
        Razon:
            Necesito llevar en history a, b y c con ese orden porque de otra forma
            si solo llevara a y b y calculo x_next el new_c no podria determinar
            en que posicion dentro de history va.
        """
        
        a, b, c = history[-3], history[-2], history[-1]
        
        f_a = self.function(a)
        f_c = self.function(c)

        if signo(f_a) == signo(f_c):
            new_a, new_b = c, b
        else:
            new_a, new_b = a, c

        history.clear()
        history.extend([new_a, new_b])

        new_c = (new_a + new_b) / 2
        return new_c

    def _get_history_size(self) -> int:
        """Retornamos 3 porque biseccion necesita a, b y c en el historial no mas"""
        return 3

class Secante(RootFinder):
    """
    Descripcion
        Clase Secante hereda de la plantilla abstracta RootFinder
        Intentar calcular la raiz de una funcion definida por el usuario
        Utiliza la sucesion estudiada del metodo de la secante
    """

    def _validate_inputs(self, a: Union[float, int], b: Union[float, int]) -> None:
        """Validamos que sean numeros diferentes"""
        if abs(a-b) < 1e-10:
            raise ValueError(f'a: {a} y b: {b} deben ser numeros distintos')
    
    def _initializate(self,a: Union[float, int], b: Union[float, int] ) -> deque:
        """
        Definimos historial necesario para generar las aproximaciones y
          calcular el error
        """
        return deque([a, b])
    
    def _iterate(self, history: deque) -> float:
        """
        Descripcion:
            Calculamos la siquiente aproximacion con la sucesion de la secante
            x_{n+1} = x_{n} - [ f(x_{n}) . (x_{n} - x_{n-1}) ] / [ f(x_{n}) - f(x_{n-1}) ]
        history:
            deque con las aproximaciones n-esima y n-1-esima
         (1)este caso no sacamos la n-1-esima porque find_root() la saca por nosotros
        """
        
        x_ant, x_curr = history[-2], history[-1]
        f_curr = self.function(x_curr)
        f_ant  = self.function(x_ant)

        denominator = f_curr - f_ant
        numerator = f_curr * (x_curr - x_ant)
        if abs(denominator) < 1e-12:
            print(f'denominator: {denominator}')
            raise ValueError(f'No se puede continuar con el metodo Secante, Division por cero') 

        x_next = x_curr - numerator / denominator
        return x_next
    
    def _get_history_size(self) -> int:
        """Calcula la cantidad de iteraciones que guardamos en history"""
        return 2

class RegulaFalsi(RootFinder):
    """
    Descripcion:
        Clase RegulaFalsi, hereda de la plantilla abstracta RootFinder
        Encuentra las raices de funciones atraves del metodo cerrado definido por la
         sucesion de regula falsi
    """
    def __init__(self, function: FunctionHandler, tol: float = 1e-7, 
                 max_iter: int = 100, check_error_x: bool = False, check_error_f: bool = True):
        super().__init__(function, tol, max_iter, check_error_x, check_error_f)

    def _validate_inputs(self, a: Union[float, int], b: Union[float, int]) -> None:
        """
        Valida que a sea menor que b
        Valida que f(a) tenga signo opuesto al de f(b)
        sino genera ValueError
        """
        if a >= b:
            raise ValueError(f'a: {a} debe ser menor que b: {b}')
        
        f_a = self.function(a)
        f_b = self.function(b)
        if signo(f_a) == signo(f_b) and signo(f_a) != 0:
            raise ValueError(f'f(a): {f_a} debe tener signo opuesto a f(b): {f_b}')

    def _initializate(self, a: Union[float, int], b: Union[float, int]) -> deque:
        """
        Define el historial de aproximaciones para generar las aproximaciones
         apartid de la sucesion de Regula Falsi
        """
        f_a = self.function(a)
        f_b = self.function(b)

        numerator = f_b * (b-a)
        denominator = f_a - f_b
        if abs(denominator) < 1e-12 and f_a > 1e-8:
            raise(ValueError(f'No se puede continuar con el metodo de regula falsi, division por cero'))
        
        c = b - numerator / denominator
        return deque([a, b, c])
    
    def _iterate(self, history: deque) -> float:
        """
        Descripcion:
            Calcula la siguiente aproximacion en el metodo de regula falsi
            apartir de las 2 aproximaciones anteriores
        history:
            deque con las aproximaciones n+1-esima, n-esimas y n-1-esimas 
            donde [x_{n-1}, x_{n}, x_{n+1}]
        Razon:
            Al igual que en biseccion en history tenemos un orden implicito
            x_{n-1} es a y x_{n} es b. 
            x_{n+1} es c la interseccion de larecta secante con el eje x
        Determinacion:
            Determina el proximo intervalo a partir de las 3 aproximaciones 
            previas y retorna la siguiente aproximacion
        Resultado:
            history: [new_a, new_b] new_c (retornado)
        
        """
        a, b, c = history[-3], history[-2], history[-1]
        
        f_a = self.function(a)
        f_c = self.function(c)

        if signo(f_a) == signo(f_c):
            new_a, new_b = c, b
            new_a_is_best = True
        else:
            new_a, new_b = a, c
            new_a_is_best = False

        history.clear()
        history.extend([new_a, new_b])
        
        if new_a_is_best:
            f_x_n = self.function(new_a)
            f_x_n_prev = self.function(new_b)
            
            denominator = f_x_n - f_x_n_prev
            if abs(denominator) < 1e-12:
                print(f'\ndenominator: {denominator}') 
                
                raise(ValueError(f'No se puede continuar con el metodo de regula falsi, division por cero'))
            
            numerator   = f_x_n * (new_a - new_b)
            new_c = new_a - numerator / denominator
        else:
            f_x_n = self.function(new_b)
            f_x_n_prev = self.function(new_a)
            
            denominator = f_x_n - f_x_n_prev
            if abs(denominator) < 1e-12:
                print(f'\ndenominator: {denominator}') 

                raise(ValueError(f'No se puede continuar con el metodo de regula falsi, division por cero'))
            
            numerator = f_x_n * (new_b - new_a)
            new_c = new_b - numerator / denominator
        return new_c
    
    def _get_history_size(self) -> int:
        """
        La cantidad de aproximaciones necesarias para 
         calcular la proxima aproximacion son 3
        """
        return 3

class RegulaFalsiModificado(RootFinder):
    """
    Descripcion:
        Clase RegulaFalsi, hereda de la plantilla abstracta RootFinder
        Encuentra las raices de funciones atraves del metodo cerrado definido por la
         sucesion de regula falsi
    """
    cant_act_a: int
    cant_act_b: int

    def __init__(self, function: FunctionHandler, tol: float = 1e-7, 
                 max_iter: int = 100, check_error_x: bool = False, check_error_f: bool = True):
        super().__init__(function, tol, max_iter, check_error_x, check_error_f)
        self.cant_act_a = 0
        self.cant_act_b = 0    

    def _validate_inputs(self, a: Union[float, int], b: Union[float, int]) -> None:
        """
        Valida que a sea menor que b
        Valida que f(a) tenga signo opuesto al de f(b)
        sino genera ValueError
        """
        if a >= b:
            raise ValueError(f'a: {a} debe ser menor que b: {b}')
        
        f_a = self.function(a)
        f_b = self.function(b)
        if signo(f_a) == signo(f_b) and signo(f_a) != 0:
            raise ValueError(f'f(a): {f_a} debe tener signo opuesto a f(b): {f_b}')

    def _initializate(self, a: Union[float, int], b: Union[float, int]) -> deque:
        """
        Define el historial de aproximaciones para generar las aproximaciones
         apartid de la sucesion de Regula Falsi
        """
        f_a = self.function(a)
        f_b = self.function(b)

        numerator = f_b * (b-a)
        denominator = f_a - f_b
        if abs(denominator) < 1e-12 and f_a > 1e-8:
            raise(ValueError(f'No se puede continuar con el metodo de regula falsi, division por cero'))
        
        c = b - numerator / denominator
        return deque([a, b, c])
    
    def _iterate(self, history: deque) -> float:
        """
        Descripcion:
            Calcula la siguiente aproximacion en el metodo de regula falsi modificado
            apartir de las 2 aproximaciones anteriores
        history:
            deque con las aproximaciones n+1-esima, n-esimas y n-1-esimas 
            donde [x_{n-1}, x_{n}, x_{n+1}]
        Razon:
            Al igual que en biseccion en history tenemos un orden implicito
            x_{n-1} es a y x_{n} es b. 
            x_{n+1} es c la interseccion de la recta secante con el eje x
        Determinacion:
            1. Determina el proximo intervalo a partir de las 3 aproximaciones 
            previas 
            2. retorna la siguiente aproximacion:
        Como:
            De [a,b,c] determino la prox actualizacion de extremos -> 
            -> (incrementar actualizado y reiniciar no actualizado) contadores
            Si al incrementar el contador, contador == 2, 
                La siguiente aproximacion se calcula con:
                    f_(otro_extremo) /= 2
        Resultado:
            history: [new_a, new_b] new_c (retornado)
        
        """
        a, b, c = history[-3], history[-2], history[-1]
        
        f_a = self.function(a)
        f_c = self.function(c)

        if signo(f_a) == signo(f_c):
            new_a, new_b = c, b
            new_a_is_best = True
            self.cant_act_b = 0
            self.cant_act_a += 1
        else:
            new_a, new_b = a, c
            new_a_is_best = False
            self.cant_act_a = 0
            self.cant_act_b += 1

        history.clear()
        history.extend([new_a, new_b])
        
        if new_a_is_best:
            f_x_n = self.function(new_a)
            f_x_n_prev = self.function(new_b)
            
            """MODIFICACION DE REGULAFALSI"""
            if self.cant_act_a >= 2:
                f_x_n_prev /= 2

            denominator = f_x_n - f_x_n_prev
            if abs(denominator) < 1e-12:
                print(f'\ndenominator: {denominator}') 
                
                raise(ValueError(f'No se puede continuar con el metodo de regula falsi, division por cero'))
            
            numerator   = f_x_n * (new_a - new_b)
            new_c = new_a - numerator / denominator
        else:
            f_x_n = self.function(new_b)
            f_x_n_prev = self.function(new_a)
            
            """MODIFICACION DE REGULAFALSI"""
            if self.cant_act_a >= 2:
                f_x_n_prev /= 2
            
            denominator = f_x_n - f_x_n_prev
            if abs(denominator) < 1e-12:
                print(f'\ndenominator: {denominator}') 

                raise(ValueError(f'No se puede continuar con el metodo de regula falsi, division por cero'))
            
            numerator = f_x_n * (new_b - new_a)
            new_c = new_b - numerator / denominator
        return new_c
    
    def _get_history_size(self) -> int:
        """
        La cantidad de aproximaciones necesarias para 
         calcular la proxima aproximacion son 3
        """
        return 3

class Halley(RootFinder):
    """
    Descripcion:
        Clase Halley hereda de la plantilla abstracta RootFinder
        Encuentra raices de funciones definidas por el usuario atraves de la
         sucesion de halley, con orden cubico
    """

    def _validate_inputs(self, x0: float) -> None:
        """Validamos que input sea un numero"""
        if not isinstance(x0, (int, float)):
            raise ValueError(f'La aproximacion inicial: {x0} debe ser un solo numero')

    def _initializate(self, x0: float) -> deque:
        """
        Define el historial de aproximaciones necesarias para el metodo de
        Halley, la primera vez no se puede calcular el error |x_0 - x_-1|
        """
        return deque([x0])

    def _iterate(self, history: deque) -> float:
        """
        Calcula la siguiente aproximacion atraves de la sucesion de Halley:
        x_{n} = 2 f(x_{n}).f'(x_{n}) / 2 f'(x_{n})² - f(x_{n}) * f''(x_{n})
        """
        x_curr = history[-1]
        f_curr = self.function(x_curr)
        f_prime_curr = self.function.f_prime(x_curr)
        f_2prime_curr = self.function.f_2prime(x_curr)

        numerator = 2 * f_curr * f_prime_curr
        denominator = 2 * f_prime_curr ** 2 - f_curr * f_2prime_curr
        
        x_next = x_curr - numerator / denominator
        return x_next 

    def _get_history_size(self) -> int:
        """
        Retorna la cantidad de aproximaciones necesarias para encontrar la 
        siguiente aproximacion y definir el error en el paso n
        """
        return 2

def signo(x: float):
    return 1 if x > 0 else 0 if x == 0 else -1

if __name__ == '__main__':
    func = FunctionHandler("cos x - x")
    solver = Halley(function=func)
    result = solver.find_root(0)
    print('='*50,'\n')
    print('CONVERGIO') if result.converged else print('NO CONVERGIO')
    print(result.to_dataframe())
    result.to_excel()
