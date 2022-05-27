import numpy as np
from os import listdir
from os.path import isfile, join


def read_compressed_deep_features(compressed_folder = "/home/felix/SequentialPatternErrorDetection/deepfeatures/BlackOak/avg_state/"):


    files = [f for f in listdir(compressed_folder) if isfile(join(compressed_folder, f))]



    for i in range(len(files)):
        c_file = compressed_folder + "out" + str(i) + ".npz"

        loaded = np.load(c_file)
        loaded_features = loaded['arr_0']

        if i == 0:
            features = loaded_features
        else:
            features = np.hstack((features, loaded_features))

    print(features.shape)

    return features