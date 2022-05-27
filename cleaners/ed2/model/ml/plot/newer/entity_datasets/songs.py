import numpy as np
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list_latex
from ml.plot.newer.column_strategy_sim.plotlatex_lib import plot_list

labels = [4]

songs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0009406177237051086, 0.0009406177237051086, 0.0014753342554172431, 0.0014753342554172431, 0.0040571670888107005, 0.0040571670888107005, 0.0040571670888107005, 0.0040571670888107005, 0.003163560695061928, 0.003163560695061928, 0.0026730449988252468, 0.0026730449988252468, 0.011338252965417591, 0.011338252965417591, 0.011338252965417591, 0.011338252965417591, 0.011338420140808728, 0.011338420140808728, 0.01132517861487757, 0.01132517861487757, 0.0011962276382607624, 0.0012193663153914128, 0.001296491370229317, 0.0013119106048679603, 0.001972690564990907, 0.0019880868901851695, 0.0020340941924299337, 0.0020340941924299337, 0.0038612414429659257, 0.0038612414429659257, 0.0038535497269440813, 0.00386752013532476, 0.0037822587464733516, 0.003774585737293445, 0.003858184808589457, 0.003858184808589457, 0.0024899304284145005, 0.00249762956453366, 0.0025207071855571955, 0.0025214457831325304, 0.0021390622104157657, 0.0021313481935665715, 0.0020854735183483053, 0.0020854735183483053, 0.0034511580252905173, 0.0034434678108943005, 0.003512325876059571, 0.003519892785760224, 0.004308752923796627, 0.0043241080565365585, 0.004324357598211778, 0.0043320355180745135, 0.00592966481668284, 0.00592966481668284, 0.005892010495143695, 0.005900067133403664, 0.006938361796936258, 0.00693073851379263, 0.00690053513573475, 0.00690053513573475, 0.014716875703185762, 0.014716875703185762, 0.014716931266871802, 0.014716986830977406, 0.015154834664228999, 0.015154834664228999, 0.015177326932011771, 0.015177326932011771, 0.011542425804104646, 0.011542425804104646, 0.011549960780152864, 0.011550092082427037, 0.01126429815923036, 0.01126429815923036, 0.011264255494828061, 0.011264255494828061, 0.01007524163117175, 0.01007524163117175, 0.010105458162188429, 0.010105496524574731, 0.009440349315745833, 0.009440349315745833, 0.009447956365959971, 0.009463098480122929, 0.007007542469375434, 0.007007542469375434, 0.006999927402499646, 0.007000007641929739, 0.006464445549477871, 0.006464445549477871, 0.006487321799466784, 0.006487321799466784, 0.004757439744220364, 0.004757439744220364, 0.004757476308690271, 0.004757476308690271, 0.004697235857360861, 0.004697235857360861, 0.004704887470930791, 0.004704887470930791, 0.004279474301987723, 0.004287154699321906, 0.004310195536638586, 0.004310195536638586, 0.004172392158674072, 0.004172392158674072, 0.00419543813460813, 0.00419543813460813, 0.004415078956072272, 0.004422753721843404, 0.00443805187213488, 0.00443805187213488, 0.004591318792875381, 0.004591318792875381, 0.004598991759498879, 0.004598991759498879, 0.003683766381393126, 0.003691458781279141, 0.003691501454636534, 0.003691501454636534, 0.004067498911867006, 0.004067498911867006, 0.00409056243307578, 0.00409056243307578, 0.004146004076647131, 0.004146004076647131, 0.004153694399062914, 0.004161384662215013, 0.00459012353285481, 0.00459012353285481, 0.004605473428035627, 0.004605473428035627, 0.004988260652014934, 0.004988260652014934, 0.00501897511296543, 0.005026672927555867, 0.004674768572000677, 0.004674768572000677, 0.00471320312981332, 0.00471320312981332, 0.004624063627115509, 0.004624063627115509, 0.004624063627115509, 0.004631752551895281, 0.004930738532180773, 0.004946108983890476, 0.004953813203491552, 0.004953813203491552, 0.00502435114974416, 0.00502435114974416, 0.00502435114974416, 0.00502435114974416, 0.004811028357311375, 0.004811028357311375, 0.004834046490112177, 0.004834046490112177, 0.0040763229713923965, 0.0040763229713923965, 0.0040917319992742965, 0.0040917319992742965, 0.004345342842147509, 0.004353044236654331, 0.004399234378858977, 0.004399234378858977, 0.004720947275041462, 0.004720947275041462, 0.004728643002881155, 0.0047363386713567225, 0.0047520963996698215, 0.0047520963996698215, 0.004752078069854004, 0.004752078069854004, 0.004919119801384756, 0.004919119801384756, 0.004926811027112881, 0.004934502193540429, 0.005287029437039263, 0.005294716073740674, 0.0053178780895715635, 0.0053178780895715635, 0.004936957368601723, 0.004944652292976434, 0.004967736709954373, 0.0049754313969005756, 0.0050059971691818785, 0.0050136912337537115, 0.005052102214440529, 0.005052102214440529, 0.004967411006980601, 0.004967411006980601, 0.0049827800981900785, 0.004998168109682021, 0.004876148151577226, 0.004876148151577226, 0.004906915201407265, 0.004930003973320886, 0.0052299681805033265, 0.0052299681805033265, 0.0052607625791621345, 0.0052607625791621345, 0.005721291377196567, 0.005728979925438443, 0.005775065441746243, 0.005790441530804609, 0.005998064891660916, 0.006005751335682182, 0.006036496519184957, 0.006036496519184957, 0.006243544483327424, 0.006243544483327424, 0.006335431013553353, 0.006381527481801472, 0.006174457326097681, 0.006174457326097681, 0.006174457326097681, 0.006189826484440641, 0.0068406882287008475, 0.006848365328320958, 0.006817866662557971, 0.006833221629033873, 0.006878384311429826, 0.006878384311429826, 0.006878384311429826, 0.006878384311429826, 0.0068014897034072925, 0.006847494964510874, 0.006870681753410307, 0.00688603382937425, 0.006795907045552609, 0.006795907045552609, 0.006803585944392435, 0.006803585944392435, 0.007514108200078529, 0.007506496006159177, 0.007537233858270748, 0.007560243282777735, 0.007918491100354755, 0.007918491100354755, 0.0079261559298035, 0.0079261559298035, 0.007910917527915227, 0.00791858250447296, 0.007972205014947885, 0.007964540480642704, 0.008043070836748753, 0.008043070836748753, 0.00805840225050317, 0.00805840225050317, 0.008319128223085004, 0.008319128223085004, 0.008349718533681688, 0.008357381969294702, 0.008127922415286037, 0.008127922415286037, 0.008150918998799299, 0.008150918998799299, 0.00938577374981263, 0.00938577374981263, 0.009431532989227146, 0.009431532989227146, 0.009653444419678883, 0.009653444419678883, 0.009661093156149244, 0.009661093156149244, 0.00886909432162582, 0.00886909432162582, 0.0088691625449035, 0.0088691625449035, 0.009173394848135332, 0.00918104890023491, 0.00918104890023491, 0.00918104890023491, 0.009347626387057844, 0.009347626387057844, 0.009363036760297034, 0.009363036760297034, 0.00988714670927794, 0.00988714670927794, 0.009894791022474543, 0.009902435276945532, 0.009820835078363042, 0.009820835078363042, 0.009874399179310637, 0.009882045568063933, 0.009480495015627822, 0.009480495015627822, 0.009480495015627822, 0.009480495015627822, 0.010924976199981574, 0.010924976199981574, 0.011000863640725458, 0.011008498192119022, 0.010835259516368077, 0.010842897019355185, 0.010858171849383182, 0.010858171849383182, 0.011609348863618923, 0.011609348863618923, 0.011624694995626354, 0.011624694995626354, 0.011715018930457899, 0.011715018930457899, 0.011715018930457899, 0.011715018930457899, 0.010800222764196416, 0.010807862777782045, 0.01085357657225156, 0.010899412783574838, 0.011309802596724533, 0.011309802596724533, 0.011309802596724533, 0.011325071789438123, 0.011552884190971863, 0.011552884190971863, 0.011598318972961562, 0.01174321503131524, 0.01225087835038893, 0.012258502514220398, 0.012258502514220398, 0.012273703590058301]

for i in range(1,len(songs)):
	labels.append((labels[-1]+10))

print labels

ranges = [labels
		  ]
list = [songs
		]
names = [
		 "Songs"
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

plot_list_latex(ranges, list, names, "Songs", x_max=200)
plot_list(ranges, list, names, "Songs")
#plot_integral(ranges, list, names, "Address", x_max=150, x_min=98)
#plot_end(ranges, list, names, "Address", x_max=150, x_min=98)
#plot_integral_latex(ranges, list, names, "Address", x_max=350)
#plot_outperform_latex(ranges, list, names, "Address",0.904, x_max=350)