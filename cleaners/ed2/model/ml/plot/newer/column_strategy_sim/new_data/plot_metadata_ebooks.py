import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform_latex

label_0 = [4, 8, 12, 16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]
fscore_metadata_with_extr_number = []
fscore_metadata_with_extr_number.append([0.0, 0.6958831341301461, 0.6958831341301461, 0.6958831341301461, 0.6958831341301461, 0.6958831341301461, 0.6958831341301461, 0.7365853658536585, 0.7377245508982035, 0.8785046728971962, 0.8819875776397516, 0.9044193216855088, 0.9106029106029105, 0.9128630705394192, 0.9147609147609148, 0.928497409326425, 0.9304257528556594, 0.9326424870466321, 0.9348500517063081, 0.940809968847352, 0.940809968847352, 0.941908713692946, 0.9430051813471502, 0.9440993788819876, 0.9473684210526316, 0.9473684210526316, 0.9495365602471678, 0.9505154639175258, 0.9524793388429752, 0.9535603715170279, 0.954639175257732, 0.9557157569515963, 0.9557157569515963, 0.9588477366255144, 0.9609856262833676, 0.9620512820512821, 0.9631147540983607, 0.9641025641025641])
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