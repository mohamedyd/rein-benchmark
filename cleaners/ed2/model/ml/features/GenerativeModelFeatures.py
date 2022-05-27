import numpy as np
from os import listdir
from os.path import isfile, join

def read_gen_mod_features(compressed_folder = "/home/felix/SequentialPatternErrorDetection/mlp_features/blackOakUppercase_loss_all/"):


    files = [f for f in listdir(compressed_folder) if isfile(join(compressed_folder, f))]

    for i in range(len(files)):
        c_file = compressed_folder + "column" + str(i) + ".npy"

        loaded = np.load(c_file)

        if i == 0:
            features = loaded
        else:
            features = np.hstack((features, loaded))

    print features.shape

    return features