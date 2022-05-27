import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list

labels = [4, 14, 24, 34, 44, 54, 64, 74, 84]

percent1 = [0.0, 0.3112454283828841, 0.2971694971694972, 0.3186480186480186, 0.21888111888111889, 0.2793206793206793, 0.23549783549783548, 0.3429570429570429, 0.29264069264069265]
percent5 = [0.0, 0.1212658776696589, 0.16193701601018368, 0.211505744493855, 0.2819914235620492, 0.33319812416854283, 0.3853494581188729, 0.43179700303788715, 0.48643990201092924]
percent10 = [0.0, 0.1834966927652604, 0.17174480989088653, 0.18074107119114022, 0.21442311520442453, 0.2498789791121252, 0.30353377108119756, 0.36657482014350407, 0.4215106255704094]
percent20 = [0.0, 0.332355816226784, 0.27067669172932335, 0.2270742358078603, 0.20911528150134048, 0.24078624078624078, 0.2485207100591716, 0.27692307692307694, 0.28158844765342955]
percent30 = [0.0, 0.4478381998386277, 0.4031311655043298, 0.3746333834442708, 0.3905517951092737, 0.39320669879382103, 0.3962857723502634, 0.40770602835826314, 0.4326264795830251]




ranges = [labels,
labels,
labels,
labels,
		  labels
		  ]
list = [percent1,
		percent5,
		percent10,
		percent20,
		percent30
		]
names = [
		 "1%",
		 "5%",
         "10%",
		 "20%",
	     "30%"
		 ]


'''
#compare round robin
ranges = [labels_optimum,
		  label_0
		  ]
list = [
		average_roundrobin_sim,
	    average_metadata_with_extr_number
		]
names = [
		 "round robin",
		 "round robin old"
		 ]
'''
'''
#vergleich random
ranges = [labels_optimum,
		  label_random
		  ]
list = [
		average_random_sim,
	    average_metadata_no_svd_random
		]
names = [
		 "round robin",
		 "round robin old"
		 ]
'''

plot_list_latex(ranges, list, names, "Address", x_max=200)
plot_list(ranges, list, names, "Address")
#plot_integral(ranges, list, names, "Address", x_max=150, x_min=98)
#plot_end(ranges, list, names, "Address", x_max=150, x_min=98)
#plot_integral_latex(ranges, list, names, "Address", x_max=350)
#plot_outperform_latex(ranges, list, names, "Address",0.904, x_max=350)