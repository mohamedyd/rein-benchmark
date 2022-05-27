import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list

labels = [4, 14, 24, 34, 44, 54, 64, 74, 84]

TypoActiveDomain = [0.0, 0.008151651708241964, 0.039828227871888444, 0.056618959389782654, 0.045446446797623705, 0.10884772506983005, 0.12322565182520304, 0.14041510790077735, 0.2039897764124788]
TypoRandom = [0.0, 0.012022310508892275, 0.05365772519712027, 0.09976305626078649, 0.08559650831497898, 0.11970896861642184, 0.13995191689799064, 0.15811186864110552, 0.19793420051295002]
TypoSwitchValue = [0.0, 0.011772471455279715, 0.039433666204868434, 0.0443282298673562, 0.0392969887632563, 0.0495655284979782, 0.07423373853781193, 0.08832694604294877, 0.10385141856995617]
TypoRemoveString = [0.0, 0.008850309135979464, 0.01606032002614291, 0.02576347771346931, 0.02855686930134064, 0.04166981421998248, 0.04661584325432498, 0.05136079419903863, 0.07174587325355834]
TypoAddString = [0.0, 0.0070257041842506, 0.5445658425930566, 0.7206812673976646, 1.0, 1.0, 1.0, 1.0, 1.0]

ranges = [labels,
labels,
labels,
labels,
		  labels
		  ]
list = [TypoActiveDomain,
		TypoRandom,
		TypoSwitchValue,
		TypoRemoveString,
		TypoAddString
		]
names = [
		 "TypoActiveDomain",
		 "TypoRandom",
         "TypoSwitchValue",
		 "TypoRemoveString",
	     "TypoAddString"
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