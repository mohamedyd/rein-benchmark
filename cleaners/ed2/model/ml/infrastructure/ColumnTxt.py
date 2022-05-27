import os


def setup_deep_learning_env(dataSet):
    dirty = dataSet.dirty_pd

    for i in range(dataSet.shape[1]):
        directories = ["/home/felix/SequentialPatternErrorDetection/data/" + dataSet.name + "/column_" + str(i) + "/orig_input",
                       "/home/felix/SequentialPatternErrorDetection/data/" + dataSet.name + "/column_" + str(i) + "/cv",
                       "/home/felix/SequentialPatternErrorDetection/data/" + dataSet.name + "/column_" + str(i) + "/features",
                       "/home/felix/SequentialPatternErrorDetection/data/" + dataSet.name + "/column_" + str(i) + "/input"
                       ]

        for dir in directories:
            if not os.path.exists(dir):
                os.makedirs(dir)


        dirty[dirty.columns[i]].to_csv("/home/felix/SequentialPatternErrorDetection/data/" + dataSet.name + "/column_" + str(i) + "/orig_input/column_" + str(i) + ".txt", index=False, header=None)


'''
from ml.flights.FlightHoloClean import FlightHoloClean
dataSet = FlightHoloClean()

dirty = dataSet.dirty_pd

for i in range(dataSet.shape[1]):
    dirty[dirty.columns[i]].to_csv("/home/felix/SequentialPatternErrorDetection/data/Flights/column_" + str(i) + ".txt", index=False, header=None)
'''

from ml.datasets.hospital import HospitalHoloClean

dataSet = HospitalHoloClean()
setup_deep_learning_env(dataSet)