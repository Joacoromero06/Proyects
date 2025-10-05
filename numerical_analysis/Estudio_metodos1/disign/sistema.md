# Diagrama de Clases del Buscador de Raíces

Arquitectura del sistema para encontrar raíces de funciones matemáticas usando los metodos numericos del programacion numerica

```mermaid
classDiagram
    %% Clase de utilidad para manejar la función matemática
    class FunctionHandler {
        -expr: sympy.Expr
        -variable: sympy.Symbol
        -_derivative: sympy.Expr
        -_second_derivative: sympy.Expr
        +__init__(expr, variable)
        +evaluate(x): float
        +derivative(): sympy.Expr
        +second_derivative(): sympy.Expr
        +is_valid_interval(a, b): bool
    }

    %% Jerarquía de las clases para encontrar raíces
    class RootFinder {
        <<abstract>>
        #function_handler: FunctionHandler
        #tolerance: float
        #max_iterations: int
        +__init__(function_handler, tolerance, max_iterations)
        +find_root(initial_guess)*: ExecutionResult
        #_iterate()*: IterationResult
        #_check_convergence(current, previous): bool
    }

    %% Implementaciones concretas de los métodos
    class Bisection {
        +find_root(interval): ExecutionResult
        #_iterate(interval): IterationResult
    }

    class Newton {
        +find_root(initial_guess): ExecutionResult
        #_iterate(current_x): IterationResult
    }

    class Secante {
        +find_root(initial_guesses): ExecutionResult
        #_iterate(x_prev, x_curr): IterationResult
    }

    class RegulaFalsi {
        +find_root(interval): ExecutionResult
        #_iterate(interval): IterationResult
    }

    class Halley {
        +find_root(initial_guess): ExecutionResult
        #_iterate(current_x): IterationResult
    }

    %% Clases para encapsular los resultados de la ejecución
    class ExecutionResult {
        +root: float
        +iterations: list~IterationResult~
        +converged: bool
        +error_message: str
        +get_root(): float
        +get_iterations_count(): int
        +get_final_error(): float
    }

    class IterationResult {
        +iteration: int
        +x_current: float
        +error: float
        +f_x: float
    }

    %% Clases de análisis y visualización
    class Comparator {
        -function_handler: FunctionHandler
        +compare_methods(results_dict): ComparisonResult
        +analyze_convergence(execution_result): AnalysisResult
    }

    class Visualizer {
        -function_handler: FunctionHandler
        +plot_function(interval)
        +plot_convergence(execution_results)
        +plot_comparison(comparison_result)
    }

    %% Relaciones
    RootFinder <|-- Bisection
    RootFinder <|-- Newton
    RootFinder <|-- Secante
    RootFinder <|-- RegulaFalsi
    RootFinder <|-- Halley

    RootFinder o-- FunctionHandler
    ExecutionResult *-- "1..*" IterationResult

    Comparator ..> ExecutionResult : uses
    Comparator ..> FunctionHandler : uses
    Visualizer ..> ExecutionResult : uses
    Visualizer ..> FunctionHandler : uses

    %% Notas explicativas
    note for FunctionHandler "Maneja la lógica matemática (evaluación, derivadas) de forma aislada."
    note for RootFinder "Define el contrato para todos los métodos de búsqueda de raíces."