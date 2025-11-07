from FunctionHandler import FunctionHandler
import NumericalMethods as nm
from Visualizer import Visualizer

func = FunctionHandler('cos x - x')
solver = nm.Newton(function=func)

viz = Visualizer(func)
viz.plot_initial()
viz.save()
result = solver.find_root(0)
viz.add_root_from_result(result)
viz.save()

