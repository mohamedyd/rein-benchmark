import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list

labels = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114]

percent1 = [0.0, 0.013212257062751262, 0.05240922522078303, 0.048119029769886426, 0.15642315666972958, 0.3015288030453923, 0.4470288121892511, 0.5693149402025179, 0.6917308192748811, 0.754713067450816, 0.8141650656739111, 0.8805628326725211]
percent5 = [0.0, 0.009402073752803466, 0.06434083387936937, 0.18153365863850884, 0.31139857688262185, 0.4223574438231896, 0.5803782870362248, 0.717420126887703, 0.8241619334902774, 0.8826590992545823, 0.9277716145881693, 0.9478356115560554]
percent10 = [0.0, 0.09758304211811758, 0.11878132675075075, 0.2604204533085631, 0.5413059386667609, 0.7310681957648681, 0.8489991947632353, 0.9069992364290675, 0.9243526335116856, 0.9542860453573226, 0.9801238044870431, 0.9868515331243449]
percent20 = [0.0, 0.3435524889834712, 0.453661800192039, 0.5508189131526386, 0.7464558689113101, 0.8424206526012498, 0.9080245597046097, 0.9407957984551117, 0.971562816660563, 0.9823136310557554, 0.9866844710966598, 0.9876086277662607]


ranges = [labels,
labels,
labels,
labels
		  ]
list = [percent1,
		percent5,
		percent10,
		percent20
		]
names = [
		 "1%",
		 "5%",
         "10%",
		 "20%"
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