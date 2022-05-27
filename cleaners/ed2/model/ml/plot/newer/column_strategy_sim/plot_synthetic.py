import numpy as np
from plotlatex_lib import plot_list_latex
from plotlatex_lib import plot_list
from plotlatex_lib import plot_integral
from plotlatex_lib import plot_integral_latex
from plotlatex_lib import plot_outperform_latex
from plotlatex_lib import plot_outperform
from plotlatex_lib import plot_end


#[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

labels_optimum = [4, 8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128]

average_001_01_opt = [0.0, 0.0, 0.0, 0.026905829596412557, 0.037854241390512293, 0.045130665771999909, 0.053519219953520471, 0.058121858555947439, 0.070839246691918961, 0.082446286191015711, 0.088410755888122014, 0.099060841786637946, 0.10547306246935431, 0.11757168323193001]
average_001_01_round = [0.0, 0.0, 0.0, 0.026905829596412557, 0.029524981150045638, 0.038703443048718325, 0.043839269910772263, 0.045965459245358824, 0.04598080875059176, 0.053569366588907229, 0.054338977799990529, 0.064793432924107663, 0.063023806154480902, 0.07578521941043323]

average_001_02_opt = [0.0, 0.0, 0.0, 0.16054503410666085, 0.1742153678163337, 0.175188264462044, 0.17800861736869286, 0.18723517157321715, 0.18712116127485837, 0.18942693706942779, 0.1887192639499303, 0.18984482431443722, 0.19068199462259514, 0.19083405040995338]
average_001_02_round = [0.0, 0.0, 0.0, 0.16054503410666085, 0.16123376011436252, 0.11867152703903208, 0.12126007977224942, 0.083813880826394183, 0.083854838905432549, 0.080477359248977101, 0.080845752592025491, 0.10537650533256877, 0.10453070277460338, 0.10527981465983556]




ranges = [labels_optimum,
		  labels_optimum
		  ]
list = [average_001_02_opt,
		average_001_02_round
		]
names = [
		 "optimum",
		 "round robin"
		 ]



plot_list(ranges, list, names, "Synthetic", x_max=200, end_of_round=28)
plot_list_latex(ranges, list, names, "Synthetic", x_max=200)
plot_integral(ranges, list, names, "Synthetic", x_max=200,x_min=28, sorted=True)
#plot_end(ranges, list, names, "Synthetic", x_max=200,x_min=28, sorted=True)
#plot_outperform(ranges, list, names, "Flights", 0.7366, x_max=200)

#plot_outperform(ranges, list, names, "Flights", 0.9, x_max=200)