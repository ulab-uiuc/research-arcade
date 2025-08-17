import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from graph_constructor.utils import figure_label_add_latex_format, figure_label_remove_latex_format


print(figure_label_remove_latex_format("\label{fig:curves-smacv2-winrate-LLM-based}"))

