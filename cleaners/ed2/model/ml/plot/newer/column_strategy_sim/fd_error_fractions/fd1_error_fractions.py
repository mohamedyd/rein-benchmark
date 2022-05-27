import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list

labels = [4, 14, 24, 34, 44, 54, 64, 74, 84]

percent1 = [0.0, 0.03636996781037662, 0.026094588554947517, 0.03522492345161249, 0.04711937383949643, 0.0569880548631202, 0.08036457734573134, 0.11353656530169232, 0.152955455688974]
percent5 = [0.0, 0.012144245354918624, 0.02930404442125612, 0.043233369632685495, 0.023958495892836114, 0.04380148423604172, 0.03935175890398375, 0.05626709958680416, 0.07981074548733363]
percent10 = [0.0, 0.008151651708241964, 0.039828227871888444, 0.056618959389782654, 0.045446446797623705, 0.10884772506983005, 0.12322565182520304, 0.14041510790077735, 0.2039897764124788]
percent20 = [0.0, 0.06899911702310257, 0.10010013615942234, 0.10781663446672116, 0.21162401267015035, 0.3480892043667392, 0.4164540247003212, 0.47065461684928567, 0.5079643980582862]
percent30 = [0.0, 0.388390322871966, 0.3830375602839889, 0.39733534396333986, 0.5319626068501038, 0.6169753118769645, 0.6722444540280308, 0.6675379525490077, 0.7067395494872226]


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