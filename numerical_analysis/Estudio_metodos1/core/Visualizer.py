import  matplotlib
matplotlib.use('Agg')
from    matplotlib.backends.backend_agg import FigureCanvasAgg
import  matplotlib.pyplot   as plt
from    matplotlib.figure   import Figure
from    matplotlib.axes     import Axes
from    typing              import Tuple, List, Optional
import  numpy               as np
import  warnings
from    scipy.optimize      import fsolve
from    FunctionHandler     import FunctionHandler
from    RootFinding         import ExecutionResult, IterationResult
class Visualizer:
    """
    Descripcion:
        Clase Visualizer para el manejo del plot adecuado de la funcion
        Encapsula los procedimientos necesarios para plotear una funcion en python
            Trabaja con numpy matplotlib
            Trabaja con scipy para determinar el rango de plotting
            Compuesta de una instancia FunctionHandler 
    Objetivos:
        Encapsular la tarea de definir el estilo de un plot adecuado
        Facilitar la semantica del sistema, modularizando la tarea
        Permite definir facilmente nuevos comportamientos acerca de la tarea
         que involucra plotear
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
        'Halley': '#34495e',
        'default':"#000000"  
    }

    def __init__(self, function_handler: FunctionHandler, figsize: Tuple[int,int] = (12, 7)) -> None:
        """
        Descripcion:
            Inicia el visualizador creando Figure, Canvas de agg y Ax
            function_handler: instancia de la funcion a plotear
            figsize: tamaño parametrizable de la figura a mostrar
        """
        self.function: FunctionHandler  = function_handler
        self.figsize: Tuple[int,int] = figsize

        # Aplicar estilo
        plt.rcParams.update(self.STYLE_CONFIG)

        # Creacion del Figure , Canvas y Axes
        self.fig = Figure(figsize=self.figsize)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax: Axes = self.fig.add_subplot(1, 1, 1)

        # Inicializacion del estado interno
        self.x_range: Optional[Tuple[float, float]] = None
        self.found_roots: List[Tuple[float, float, str]] = []

    def _find_suitable_interval(self, 
        search_range: Tuple[float, float] = (-10, 10),
        n_attempts: int = 10
    ) -> Tuple[float, float]:
        """
        Descripcion:
            Encuentra un intervalo donde la funcion cruza el eje x
        Estrategia:
            1. Intentar n_attempts veces con puntos 'cualesquiera'
                encontrar una raiz con fsolve de scipy
            2. Si encontramos en una raiz, terminamos y retornamos
                un intervalo centrado en la raiz
            3. Si no encontramos una raiz, terminamos y retornamos 
                el intervalo donde buscamos search_range
        """
        a, b = search_range
        test_points = np.linspace(a, b, n_attempts)

        found_roots = []

        # Manejo de advertencias y errores, del sistema
        with warnings.catch_warnings():
            """
            * El bloque with, es una forma de gestionar recursos que deben ser 
            * configurados y liberados, incluso si hay errores. 
            * Son llamados ContextManagers, instancias de objetos que implementen 
            * __enter__() y __exit__()
            """
            warnings.simplefilter("ignore")

            for x0 in test_points:
                try: 
                    # Intentar encontrar una raiz con fsolve
                    root, info, ier, msg     = fsolve(
                                    func=self.function,
                                    x0=x0,
                                    full_output=True
                                )  

                    if ier == 1 and abs(self.function(root[0])) < 1e-3:
                        found_roots.append(root[0])
                except(ValueError, RuntimeError, Warning):
                    # No paramos el sistema si fsolve da error
                    continue
        
        # Si encontramos raices, centramos grafico
        if found_roots:
            margin = 5
            x_min = found_roots[0] - margin
            x_max = found_roots[0] + margin
        else:
            x_min = a
            x_max = b
        
        return (x_min, x_max)

    def plot_initial(self,
            x_range: Optional[Tuple[float, float]] = None,
            n_points: int = 1000,
            show_grid: bool = True,
            title: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Descripcion:
            GEnera el grafico inicial de la funcion
            Crea las instancias self.ax y self.fig
            Seteamos sus propiedades y retornamos dichas fig, ax
        Args:
            x_range: es el rango de abcisas donde plotear
            n_points: es la 'densidad' de puntos
            show_grid: bandera para mostrar la cuadricula
            title: titulo para el grafico
        Returns:
            La figura y el axe, listos para plotear x contra y
        """

        # Determinar intervalo
        if x_range is None:
            x_range = self._find_suitable_interval()
        
        # Establecer rango de abscisas
        self.x_range = x_range
        x_min, x_max = x_range

        # Generar puntos para graficar
        x = np.linspace(x_min, x_max, n_points)

        # Evaluar la funcion, manejando los errores posibles
        y = np.zeros_like(x)
        for i, xi in enumerate(x):
            try:
                y[i] = self.function(xi)
            except(ValueError, ZeroDivisionError, OverflowError):
                y[i] = np.nan

        #print(f'x: {len(x)} y: {len(y)}')
        #print(x,y)
        # Graficar funcion
        self.ax.plot(x, y, color= '#2c3e50', linewidth=2.5, label='$f(x)$', zorder=2)

        # Pintar eje x
        self.ax.axhline(y=0, color='#7f8c8d' , linestyle='--',linewidth=1.5, zorder=2)

        # Pintar eje y
        if x_min <= 0 <= x_max:
            self.ax.axvline(x=0, color='#7f8c8d' , linestyle='--',linewidth=1.5, zorder=2)

        # Configurar grilla
        if show_grid:
            self.ax.grid(visible=True, linestyle=':', alpha=0.6, zorder=0 )

        # Configurar etiquetas
        self.ax.set_xlabel('$x$', fontsize=13, fontweight='bold')
        self.ax.set_ylabel('$f(x)$', fontsize=13, fontweight='bold')

        # Configuarar tiulo
        if title is None:
            # Manejar los errores
            try:
                my_title = 'f(x) = '+self.function.latex_str()
            except:
                my_title = 'f(x) = '+self.function.expr_str
        self.ax.set_title(my_title, fontsize=15, fontweight='bold', pad=20)

        # Configurar leyenda
        self.ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, framealpha=0.95)

        # Ajustar los limits del eje y
        y_valid = y[~np.isnan(y)]
        if len(y_valid) > 0:
            dist = y_valid.max() - y_valid.min()
            margin = dist / 10
            self.ax.set_ylim(bottom=y_valid.min()-margin, top=y_valid.max()+margin)

        # Optimizar el espacion entre la figura y los ejes
        self.fig.tight_layout()

        return self.fig, self.ax

    def add_root(self, root: float, f_root: float, method_name: str) -> None:
        """
        Descripcion:
            Modulo para agregar al axes una raiz encontrado.
            Encapsula la tarea de plotear correctamente una raiz
        Args:
            root: El valor de la raiz
            f_root: su imagen atraves de f (casi 0)
            method_name: nombre del metodo con el que se encontro
        """

        # Chequeo si hay un plot inicial
        if self.ax is None:
            raise RuntimeError('Debe llamar a plot_initial() antes de llamar add_root()')

        # Obtengo el color segun el metodo
        color = self.METHOD_COLORS.get(method_name, self.METHOD_COLORS['default'])

        # Graficar punto
        self.ax.plot(root, f_root, marker='o', markersize=12, color=color,
                     markeredgecolor='white',markeredgewidth=2,
                     label=f'{method_name}: $x ≈ {root:.6f}$', zorder=5)
        
        # Guardar raiz en el registro
        self.found_roots.append( (root, f_root, method_name) )

        # Actualizar leyenda
        self.ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, framealpha=0.95)
        self.fig.canvas.draw()

    def add_root_from_result(self, result: ExecutionResult) -> None:
        """
        Descripcion:
            Agrega una raiz al plot, desde un resultado de un metodo
             solo si hay convergencia
        """
        if not result.converged:
            print(f'Error al agregar raiz: {result.method_name} no convergio')
            return
        
        root = result.root if result.root is not None else 0
        f_root = self.function(root)
        method_name = result.method_name    
        
        self.add_root(root, f_root, method_name) 
        print(f'Raiz de {method_name} agregada al grafico.')
    
    def compare_methos(self, result_list: List[ExecutionResult]) -> None:
        """
        Descripcion:
            Comparativa visual del resultado de los metodos en el grafico
        """
        for result in result_list:
            self.add_root_from_result(result)
            self._plot_iteration_paths(result)
    
    def _plot_iteration_paths(self, result: ExecutionResult) -> None:
        """
        Dibuja en la figura la evolucion de las aproximaciones
        """
        return

    def save(self, filename: str = 'root_finding.png', dpi: int = 300) -> str:
        """
        Guarda el grafico en un archivo
        Args:
            filename: Nombre del archivo
            dpi: dot per inch
        """
        if self.fig is None:
            raise RuntimeError(f'No hay grafico para guardar')
        self.fig.savefig(fname=filename, dpi=dpi, bbox_inches='tight', facecolor=self.fig.get_facecolor())
        
        print('Grafico guardado correctamente')
        return filename


    






