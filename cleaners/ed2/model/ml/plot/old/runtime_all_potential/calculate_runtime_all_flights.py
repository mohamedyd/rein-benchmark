from ml.datasets.flights import FlightHoloClean
from ml.plot.old.runtime_all_potential import PlotterLatex

data = FlightHoloClean()


real_time = [1.671288013458252, 2.7181499004364014, 3.7631049156188965, 4.771069049835205, 6.065839052200317, 7.303418874740601, 8.573493003845215, 10.23646593093872, 11.149570941925049, 12.078059911727905, 12.956594944000244, 14.053132057189941, 14.972596883773804, 15.886981964111328, 16.90476393699646, 17.824955940246582, 18.75483989715576, 19.767791032791138, 20.81743097305298, 21.76949191093445, 22.759278059005737, 23.78637409210205, 24.67376708984375, 25.761350870132446, 26.837049961090088, 27.9605770111084, 29.0759220123291, 30.228930950164795, 31.439960956573486, 32.43942308425903, 33.674745082855225]


fscore_metadata_no_svd_absolute_potential = []
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30257029498016214, 0.56367828563678291, 0.69897659098929543, 0.81845732720696052, 0.82742636407532599, 0.82714023742881959, 0.84243676803464229, 0.84110191683251279, 0.8580651879620953, 0.8661815881050301, 0.87795234626989915, 0.88092794087456383, 0.88004136504653563, 0.8945514812512948, 0.90140551795939616, 0.89645861460172394, 0.90111030403652581, 0.90158135663753647, 0.9171555006689307, 0.91854134565998968, 0.92624215615677397, 0.93310147299508994, 0.93340163934426235, 0.93449334698055264, 0.93998172032090999, 0.94219417870954603, 0.9442755825734549])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.53216052776250689, 0.67792521109770798, 0.78574265289912637, 0.80781250000000004, 0.80839433870180577, 0.83740591813589027, 0.83552562348731974, 0.84651261394954425, 0.83897391679241218, 0.85497470489038785, 0.85799573560767595, 0.88928496540155177, 0.89705730511099624, 0.89944249432170142, 0.90283359093250903, 0.90837965822524191, 0.91550451189499593, 0.91695786228160336, 0.92139425534089747, 0.92155057788687744, 0.92498727735368946, 0.92740619902120724, 0.9281272300948108, 0.93702464253118345, 0.93821649589124467, 0.94104492783085991])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.49752229930624381, 0.63269325455440661, 0.76299034709385916, 0.76323348379154698, 0.74785855479903363, 0.7848408064338438, 0.79439861050803307, 0.82181818181818178, 0.85786494136676117, 0.86429081997345036, 0.8678897202008814, 0.87148553317656674, 0.8725339875293876, 0.88868388683886834, 0.89334850522395792, 0.89643221202854229, 0.90147225368063422, 0.90579561744828996, 0.91599678456591638, 0.92004845548152636, 0.91963746223564957, 0.92260185277418072, 0.92825067358547042, 0.93227810058104588, 0.93560415197017033, 0.94184995480566436])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.57879139286174464, 0.70524412296564187, 0.81956352299298529, 0.81321725366046704, 0.82512733446519526, 0.82565330141631754, 0.84501441495178442, 0.82026313063296385, 0.83577202496674496, 0.84453482460599905, 0.84446700507614214, 0.84812339331619546, 0.88476426799007446, 0.89269294727575066, 0.89381311893322879, 0.89889196675900274, 0.90198793393333998, 0.90889263408347387, 0.90936136453788163, 0.91906367784169807, 0.92921745969327563, 0.93591535236574164, 0.93590255591054317, 0.93741864423750865, 0.94031644302022832, 0.94554207200882556])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.30139823925427239, 0.49101624263331889, 0.61994076999012837, 0.73693015474696777, 0.77364463791056592, 0.79677038204017325, 0.81815406533021273, 0.81730476287194076, 0.82406105643703564, 0.84589357120885889, 0.84621676891615538, 0.84672725416113548, 0.85363151002316917, 0.87933785400877529, 0.88781470292044307, 0.89216571371250875, 0.89891041162227603, 0.89915881220229044, 0.90515840779853785, 0.91507237574653311, 0.91525080906148859, 0.91640617122113543, 0.92020683362060218, 0.93954148914587143, 0.9402162823913488, 0.94207627982867626, 0.94268415741675071])

end_time_in_minutes = 10

dboost_fscore = 0.53697
runtime_dboost_sec = 13

nadeef_fscore = [0.0, 0.0558375634518, 0.0623663578047, 0.0947314226875, 0.0947314226875, 0.274587767513, 0.286835613366, 0.300510090929, 0.3, 0.340531561462, 0.387727708533, 0.485119047619, 0.491457461646, 0.575188886181, 0.590985509567, 0.590985509567]
nadeef_time = [0, 5.21124005318, 8.08482217789, 11.8395302296, 15.646679163, 19.3662250042, 22.8719530106, 26.3522469997, 29.7425701618, 33.2708342075, 36.9516701698, 40.7674210072, 44.4073810577, 82.1618771553, 122.943278074, end_time_in_minutes * 60]
openrefine_fscore = 0.7366
openrefine_time = 3


'''
dboost_sizes = [0, 10, 10, 20, 30, 40, 50]
dboost_times_list = [0.0, 13.65, 13.655723571777344, 14.224757432937622, 17.99796633720398, 18.472256469726563, 19.212850236892699]
dboost_fscore_list = [0.0, 0.0, 0.50084586464033731, 0.5043579418063755, 0.50731545624338281, 0.47764829404682374, 0.5012648968083292]
Plotter2(data, real_time, fscore_0,
        dboost_times_list, dboost_fscore_list,dboost_sizes,
        nadeef_time, nadeef_fscore,
        openrefine_time, openrefine_fscore, 4, 12*60)
'''

'''
dboost_sizes = [0, 10, 10, 20, 30, 40, 50]
dboost_times_list = [0.0, 13.65, 13.655723571777344, 14.224757432937622, 17.99796633720398, 18.472256469726563, 19.212850236892699]
dboost_fscore_list = [0.0, 0.0, 0.50084586464033731, 0.5043579418063755, 0.50731545624338281, 0.47764829404682374, 0.5012648968083292]
Plotter2(data, real_time, fscore_0,
        dboost_times_list, dboost_fscore_list,dboost_sizes,
        nadeef_time, nadeef_fscore,
        openrefine_time, openrefine_fscore, 4, 12*60)
'''

'''
# gaussian
dboost_sizes = [0, 36, 36]
dboost_times_list = [0.0, 16.85, 16.85497326850891]
dboost_fscore_list = [0.0, 0.0, 0.44768652437758655]
'''

#histogram
#[10, 20, 30, 40, 50]
#[6.7065493583679201, 6.9697505950927736, 7.4651740074157713, 7.7677943706512451, 8.2837226390838623]
#[0.54035752907803958, 0.55280283795382656, 0.5538482738620939, 0.57204815754700777, 0.55900313425687165]



#rows = 10, gaussian
dboost_gaussian_10_sizes = [0, 10, 10, 0]
dboost_gaussian_10_times_list = [0.0, 13.13, 13.137000608444215, end_time_in_minutes * 60]
dboost_gaussian_10_fscore_list =[0.0, 0.0, 0.44705223812812456, 0.44705223812812456]

rows = 216.0 / data.shape[1]
dboost_gaussian_max_sizes = [0, rows, rows, 0]
dboost_gaussian_max_times_list = [0.0, 15.48, 15.485503673553467, end_time_in_minutes*60]
dboost_gaussian_max_fscore_list =[0.0, 0.0, 0.47058360656956266, 0.47058360656956266]



'''
Plotter2(data, real_time, fscore_0,
        dboost_gaus_times_list, dboost_gaus_fscore_list,dboost_gaus_sizes,
        dboost_hist_times_list, dboost_hist_fscore_list,dboost_hist_sizes,
        nadeef_time, nadeef_fscore,
        openrefine_time, openrefine_fscore, 8, 10*60)
'''


dboost_hist_10_sizes = [0, 10, 10, 0]
dboost_hist_10_times_list = [0.0, 7.89, 7.8921535968780514, end_time_in_minutes*60]
dboost_hist_10_fscore_list =[0.0, 0.0, 0.5677847401400663, 0.5677847401400663]

rows = 216.0 / data.shape[1]
dboost_hist_max_sizes = [0, rows, rows, 0]
dboost_hist_max_times_list = [0.0, 9.12, 9.1266319274902337, end_time_in_minutes*60]
dboost_hist_max_fscore_list =[0.0, 0.0, 0.56644757565831183, 0.56644757565831183]


dboost_mixture_10_sizes = [0, 10, 10, 0]
dboost_mixture_10_times_list = [0.0, 60.28, 60.289866638183597, end_time_in_minutes*60]
dboost_mixture_10_fscore_list =[0.0, 0.0, 0.5278687132968265, 0.5278687132968265]

rows = 216.0 / data.shape[1]
dboost_mixture_max_sizes = [0, rows, rows, 0]
dboost_mixture_max_times_list = [0.0, 71.54, 71.548294544219971, end_time_in_minutes*60]
dboost_mixture_max_fscore_list =[0.0, 0.0, 0.65656190394490266, 0.65656190394490266]


PlotterLatex(data, real_time, fscore_metadata_no_svd_absolute_potential,
         [dboost_gaussian_10_times_list, dboost_gaussian_max_times_list, dboost_hist_10_times_list, dboost_hist_max_times_list, dboost_mixture_10_times_list, dboost_mixture_max_times_list],
         [dboost_gaussian_10_fscore_list, dboost_gaussian_max_fscore_list, dboost_hist_10_fscore_list, dboost_hist_max_fscore_list, dboost_mixture_10_fscore_list, dboost_mixture_max_fscore_list],
         [dboost_gaussian_10_sizes, dboost_gaussian_max_sizes, dboost_hist_10_sizes, dboost_hist_max_sizes, dboost_mixture_10_sizes, dboost_mixture_max_sizes],
         ["dBoost Gaussian on 10 rows", "dBoost Gaussian on max rows", "dBoost Histogram on 10 rows", "dBoost Histogram on max rows", "dBoost Mixture on 10 rows", "dBoost Mixture on max rows"],
         nadeef_time, nadeef_fscore,
         openrefine_time, openrefine_fscore, None, end_time=end_time_in_minutes*60, filename="Flights")