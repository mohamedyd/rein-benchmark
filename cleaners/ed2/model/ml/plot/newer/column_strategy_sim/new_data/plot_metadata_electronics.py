import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform_latex

label_0 = [4, 14, 24, 34, 44]
fscore_metadata_with_extr_number = []
fscore_metadata_with_extr_number.append([0.5065502183406114, 0.9913043478260869, 0.9970845481049563, 0.9970845481049563, 1.0])
average_metadata_with_extr_number = list(np.mean(np.matrix(fscore_metadata_with_extr_number), axis=0).A1)




#compare round robin
ranges = [label_0
		  ]
list = [
		average_metadata_with_extr_number
		]
names = [
		 "ED"
		 ]


plot_list_latex(ranges, list, names, "Products", x_max=200)
plot_list(ranges, list, names, "Products", x_max=150)