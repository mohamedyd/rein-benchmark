from ml.datasets.mohammad import MohammadDataSet
from ml.plot.old.runtime_all import Plotter

data = MohammadDataSet("bikes", 30, 0, 20)


real_time = [7.784142971038818, 11.431941986083984, 14.962963104248047, 18.544862031936646, 20.023201942443848, 22.145452976226807, 27.308592081069946, 29.698084115982056, 31.3632071018219, 33.012107133865356, 34.51948094367981, 36.12201809883118]

fscore_0 = []
fscore_0.append([0.0, 0.0, 0.0, 0.045098039215686274, 0.66084620550705164, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_0.append([0.0, 0.0, 0.0, 0.0062305295950155761, 0.10058027079303676, 0.70959264126149801, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_0.append([0.0, 0.0, 0.0, 0.0082987551867219917, 0.014477766287487074, 0.67950481430536447, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_0.append([0.0, 0.0, 0.0, 0.027477919528949953, 0.027477919528949953, 0.67311679336558394, 0.9979123173277662, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_0.append([0.0, 0.0, 0.0, 0.034205231388329982, 0.1049618320610687, 0.70299003322259135, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

dboost_fscore = 0.0
runtime_dboost_sec = 22

nadeef_fscore = [0.0]
nadeef_time = [0]

openrefine_fscore = 0.0
openrefine_time = 0

Plotter(data, real_time, fscore_0,
        runtime_dboost_sec, dboost_fscore,
        nadeef_time, nadeef_fscore,
        openrefine_time, openrefine_fscore, 8)