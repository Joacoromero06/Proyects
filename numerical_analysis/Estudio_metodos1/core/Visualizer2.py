import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.optimize import brentq, fsolve
from typing import Optional, Tuple, List
from dataclasses import dataclass
import warnings

class Visualizer:
    """
    Clase para plotear los metodos
    
    Como?
    1. Encuentra automáticamente un intervalo adecuado usando scipy
    2. Genera el plot
    3. Permite agregar las raices que enontremos de NUmerical Methods
    4. Puede plotear las raices de varios metodos, distinguiendolos
    """
    
    # Configuración de estilo moderno
    STYLE_CONFIG = {
        'figure.facecolor': '#f8f9fa',
        'axes.facecolor': '#ffffff',
        'axes.edgecolor': '#dee2e6',
        'axes.labelcolor': '#212529',
        'text.color': '#212529',
        'xtick.color': '#495057',
        'ytick.color': '#495057',
        'grid.color': '#e9ecef',
        'grid.alpha': 0.6,
        'lines.linewidth': 2.5,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    }
    
    # Paleta de colores para diferentes métodos
    METHOD_COLORS = {
        'Newton': '#e74c3c',       
        'Secante': '#3498db',      
        'Biseccion': '#2ecc71',    
        'RegulaFalsi': '#f39c12',  
        'RegFalsiMod': '#9b59b6',  
        'Halley': '#34495e'        
    }
    
    def __init__(self, function_handler, figsize: Tuple[int, int] = (12, 7)):
        """
        Descripcion:
            Inicializa el visualizador.
            function_handler: Objeto FunctionHandler con la función simbólica
            figsize: Tamaño de la figura (ancho, alto) en pulgadas
        """
        self.function = function_handler
        self.figsize = figsize
        
        # Estado interno
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.x_range: Optional[Tuple[float, float]] = None
        self.found_roots: List[Tuple[float, float, str]] = []  # (x, f(x), método)
        
        # Aplicar estilo
        plt.rcParams.update(self.STYLE_CONFIG)
    
   
    def _find_suitable_interval(
        self, 
        search_range: Tuple[float, float] = (-10, 10),
        n_attempts: int = 5
    ) -> Tuple[float, float]:
        """
        Encuentra automáticamente un intervalo donde la función cruza el eje x.
        
        Estrategia:
        1. Intenta encontrar una raíz con scipy.fsolve en varios puntos iniciales
        2. Si encuentra una, expande el intervalo alrededor de ella
        3. Si no encuentra ninguna, usa el rango de búsqueda original
        
        Args:
            search_range: Rango de búsqueda inicial
            n_attempts: Número de intentos con diferentes puntos iniciales
        
        Returns:
            Tupla (x_min, x_max) del intervalo a graficar
        """
        a, b = search_range
        x_test_points = np.linspace(a, b, n_attempts)
        
        roots_found = []
        
        # Intentar encontrar raíces desde diferentes puntos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for x0 in x_test_points:
                try:
                    # Intentar encontrar raíz con fsolve
                    root, info, ier, msg = fsolve(
                        self.function, 
                        x0, 
                        full_output=True
                    )
                    
                    # Verificar si realmente es una raíz
                    if ier == 1 and abs(self.function(root[0])) < 1e-3:
                        roots_found.append(root[0])
                
                except (ValueError, RuntimeError, Warning):
                    continue
        
        # Si encontró raíces, centrar el gráfico en la primera
        if roots_found:
            root = roots_found[0]
            # Crear intervalo centrado en la raíz
            margin = 5  # Margen a cada lado
            x_min = root - margin
            x_max = root + margin
            
            print(f"✓ Raíz encontrada aproximadamente en x ≈ {root:.4f}")
            print(f"  Graficando intervalo: [{x_min:.2f}, {x_max:.2f}]")
        else:
            # No encontró raíces, usar rango completo
            x_min, x_max = search_range
            print(f"⚠ No se encontró raíz automáticamente.")
            print(f"  Graficando intervalo completo: [{x_min}, {x_max}]")
        
        return (x_min, x_max)
    
    # ========================================================================
    # MÉTODO 2: Plot inicial de la función
    # ========================================================================
    
    def plot_initial(
        self, 
        x_range: Optional[Tuple[float, float]] = None,
        n_points: int = 1000,
        show_grid: bool = True,
        title: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Genera el gráfico inicial de la función.
        
        Args:
            x_range: Intervalo a graficar. Si None, se determina automáticamente.
            n_points: Número de puntos para graficar
            show_grid: Si True, muestra la grilla
            title: Título personalizado. Si None, genera automáticamente.
        
        Returns:
            Tupla (figura, ejes) de matplotlib
        """
        # Determinar intervalo
        if x_range is None:
            x_range = self._find_suitable_interval()
        
        self.x_range = x_range
        x_min, x_max = x_range
        
        # Crear figura y ejes
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Generar puntos para graficar
        x = np.linspace(x_min, x_max, n_points)
        
        # Evaluar función (con manejo de errores)
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            try:
                y[i] = self.function(xi)
            except (ValueError, ZeroDivisionError, OverflowError):
                y[i] = np.nan
        
        # Graficar función
        self.ax.plot(
            x, y, 
            color='#2c3e50', 
            linewidth=2.5, 
            label=f'$f(x)$',
            zorder=2
        )
        
        # Eje x (y=0)
        self.ax.axhline(
            y=0, 
            color='#7f8c8d', 
            linestyle='--', 
            linewidth=1.5, 
            alpha=0.7,
            zorder=1
        )
        
        # Eje y (x=0) si está en el rango
        if x_min <= 0 <= x_max:
            self.ax.axvline(
                x=0, 
                color='#7f8c8d', 
                linestyle='--', 
                linewidth=1.5, 
                alpha=0.7,
                zorder=1
            )
        
        # Configuración de la grilla
        if show_grid:
            self.ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
        
        # Etiquetas y título
        self.ax.set_xlabel('$x$', fontsize=13, fontweight='bold')
        self.ax.set_ylabel('$f(x)$', fontsize=13, fontweight='bold')
        
        if title is None:
            # Intentar obtener la expresión simbólica
            try:
                expr_str = str(self.function.expr)
                title = f'Gráfico de $f(x) = {expr_str}$'
            except:
                title = 'Gráfico de la Función'
        
        self.ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Leyenda
        self.ax.legend(
            loc='best', 
            frameon=True, 
            shadow=True, 
            fancybox=True,
            framealpha=0.95
        )
        
        # Ajustar límites del eje y para mejor visualización
        y_valid = y[~np.isnan(y)]
        if len(y_valid) > 0:
            y_margin = (y_valid.max() - y_valid.min()) * 0.1
            self.ax.set_ylim(y_valid.min() - y_margin, y_valid.max() + y_margin)
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    # ========================================================================
    # MÉTODO 3: Agregar raíces encontradas al gráfico
    # ========================================================================
    
    def add_root(
        self, 
        root: float, 
        f_root: float,
        method_name: str,
        marker: str = 'o',
        marker_size: int = 12,
        show_annotation: bool = True
    ) -> None:
        """
        Agrega una raíz encontrada al gráfico existente.
        
        Args:
            root: Valor de x donde se encontró la raíz
            f_root: Valor de f(x) en la raíz (debería ser ≈ 0)
            method_name: Nombre del método que encontró la raíz
            marker: Estilo del marcador ('o', 's', '^', etc.)
            marker_size: Tamaño del marcador
            show_annotation: Si True, muestra anotación con coordenadas
        """
        if self.ax is None:
            raise RuntimeError("Debe llamar plot_initial() antes de add_root().")
        
        # Obtener color según el método
        color = self.METHOD_COLORS.get(method_name, self.METHOD_COLORS['default'])
        
        # Graficar punto
        self.ax.plot(
            root, f_root,
            marker=marker,
            markersize=marker_size,
            color=color,
            markeredgecolor='white',
            markeredgewidth=2,
            label=f'{method_name}: $x ≈ {root:.6f}$',
            zorder=5
        )
        
        # Agregar anotación
        if show_annotation:
            # Determinar posición de la anotación
            y_range = self.ax.get_ylim()
            annotation_offset = (y_range[1] - y_range[0]) * 0.05
            
            self.ax.annotate(
                f'$x = {root:.6f}$\n$f(x) = {f_root:.2e}$',
                xy=(root, f_root),
                xytext=(root, f_root + annotation_offset),
                fontsize=9,
                ha='center',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=color,
                    alpha=0.2,
                    edgecolor=color
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0',
                    color=color,
                    lw=1.5
                ),
                zorder=6
            )
        
        # Guardar raíz en el registro
        self.found_roots.append((root, f_root, method_name))
        
        # Actualizar leyenda
        self.ax.legend(
            loc='best',
            frameon=True,
            shadow=True,
            fancybox=True,
            framealpha=0.95
        )
        
        # Redibujar
        self.fig.canvas.draw_idle()
    
    # ========================================================================
    # MÉTODO 4: Agregar múltiples raíces desde ExecutionResult
    # ========================================================================
    
    def add_root_from_result(self, result) -> None:
        """
        Agrega una raíz desde un objeto ExecutionResult.
        
        Args:
            result: Objeto ExecutionResult con la raíz encontrada
        """
        if not result.converged:
            print(f"⚠ {result.method_name} no convergió. No se agregará al gráfico.")
            return
        
        root = result.root
        f_root = self.function(root)
        
        self.add_root(root, f_root, result.method_name)
        
        print(f"✓ Raíz de {result.method_name} agregada al gráfico: x = {root:.8f}")
    
    # ========================================================================
    # MÉTODO 5: Comparar múltiples métodos visualmente
    # ========================================================================
    
    def compare_methods(
        self, 
        results: List,  # List[ExecutionResult]
        show_iterations: bool = False
    ) -> None:
        """
        Compara visualmente múltiples métodos en el mismo gráfico.
        
        Args:
            results: Lista de objetos ExecutionResult
            show_iterations: Si True, muestra la trayectoria de iteraciones
        """
        for result in results:
            if result.converged:
                self.add_root_from_result(result)
                
                # Opcionalmente mostrar trayectoria de iteraciones
                if show_iterations:
                    self._plot_iteration_path(result)
    
    def _plot_iteration_path(self, result) -> None:
        """
        Dibuja la trayectoria de iteraciones de un método.
        """
        if not result.iterations:
            return
        
        x_values = [it.x_n for it in result.iterations]
        y_values = [it.f_x_n for it in result.iterations]
        
        color = self.METHOD_COLORS.get(result.method_name, self.METHOD_COLORS['default'])
        
        # Línea conectando iteraciones
        self.ax.plot(
            x_values, y_values,
            linestyle=':',
            linewidth=1,
            color=color,
            alpha=0.5,
            zorder=3
        )
        
        # Puntos de iteraciones
        self.ax.scatter(
            x_values[:-1], y_values[:-1],
            s=30,
            color=color,
            alpha=0.3,
            zorder=4
        )
    
    # ========================================================================
    # MÉTODO 6: Guardar figura
    # ========================================================================
    
    def save(
        self, 
        filename: str = 'root_finding_plot.png',
        dpi: int = 300,
        transparent: bool = False
    ) -> str:
        """
        Guarda el gráfico en un archivo.
        
        Args:
            filename: Nombre del archivo
            dpi: Resolución (dots per inch)
            transparent: Si True, fondo transparente
        
        Returns:
            Ruta del archivo guardado
        """
        if self.fig is None:
            raise RuntimeError("No hay gráfico para guardar.")
        
        self.fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches='tight',
            transparent=transparent,
            facecolor=self.fig.get_facecolor()
        )
        
        print(f"✓ Gráfico guardado en: {filename}")
        return filename
    
    # ========================================================================
    # MÉTODO 7: Mostrar gráfico
    # ========================================================================
    
    def show(self) -> None:
        """Muestra el gráfico en una ventana interactiva."""
        if self.fig is None:
            raise RuntimeError("No hay gráfico para mostrar.")
        
        plt.show()
    
    # ========================================================================
    # MÉTODO 8: Resetear visualizador
    # ========================================================================
    
    def reset(self) -> None:
        """Resetea el visualizador para crear un nuevo gráfico."""
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig = None
        self.ax = None
        self.x_range = None
        self.found_roots = []
        
        print("✓ Visualizador reseteado.")