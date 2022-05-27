import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list

labels = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]

percent1 = [0.0, 0.3131404259281497, 0.363028971028971, 0.37383727383727383, 0.30779220779220784, 0.4387945387945388, 0.3956127206127206, 0.34080086580086577, 0.42862137862137867, 0.3491341991341991]
percent5 = [0.0, 0.12725294959416703, 0.15866200901632738, 0.22126148633239745, 0.3147271443378923, 0.36877859439190025, 0.41058938963978325, 0.4664317772624288, 0.4932271592721813, 0.5282520600698637]
percent10 = [0.0, 0.19194443769022176, 0.1764088053517086, 0.17030337311430116, 0.2035824581488567, 0.24296474348479297, 0.3005536022266852, 0.356942164163595, 0.41144723863239524, 0.4611879425824189]
percent20 = [0.0, 0.3085154131452218, 0.27074237382735833, 0.251753004670542, 0.24671690407851127, 0.24089711278176423, 0.2479422016439468, 0.2644119445803447, 0.2970386769694091, 0.3352283076423682]
percent30 = [0.0, 0.4516952205089678, 0.42087374644656006, 0.39561620970693856, 0.3919595686045375, 0.3888065700140844, 0.40260206707501345, 0.415200191206967, 0.4387787307231835, 0.467153738772462]



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