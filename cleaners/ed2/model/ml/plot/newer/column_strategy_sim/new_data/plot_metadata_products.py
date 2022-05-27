import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_integral_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_outperform_latex

label_0 = [4, 8, 12, 16, 20, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164, 174, 184, 194, 204, 214, 224, 234, 244, 254, 264, 274, 284, 294, 304, 314, 324]
fscore_metadata_with_extr_number = []
fscore_metadata_with_extr_number.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.331457345971564, 0.33732826658871673, 0.5577788894447223, 0.7119571162892462, 0.8060665362035224, 0.9256198347107438, 0.9256198347107438, 0.9131135793622357, 0.9255674819637516, 0.9286089238845144, 0.9271662149164991, 0.9275843974112297, 0.9279090113735784, 0.9296244296244296, 0.9296861301069613, 0.9298491757278148, 0.9302570400912361, 0.9307922009485334, 0.9330871361997713, 0.9324526640246588, 0.9327464788732395, 0.9332042594385286, 0.933392148213498, 0.935005298481102, 0.9333333333333332, 0.9338404387051124, 0.9346174235730694, 0.9336518046709131, 0.9345827439886846, 0.934488550968084])
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