import numpy as np
from os import listdir
from os.path import isfile, join
from optparse import OptionParser


def read_all_deep_features(data_path, column_id, text_file):
    column_folder = data_path + "/column_" + str(column_id) + "/features/"
    files = [f for f in listdir(column_folder) if isfile(join(column_folder, f))]

    for i in range(len(files)):
        path = column_folder + 'features' + str(i) + '.npy'

        if i == 0:
            matrix = np.load(path)
        else:
            matrix = np.vstack((matrix, np.load(path)))
        print matrix.shape

    size_list = []
    with open(text_file, 'r') as f:
        for line in f:
            size_list.append(len(line) - 1)
    f.closed

    return matrix, size_list

def deep_features_last_state(data_path="/home/felix/SequentialPatternErrorDetection/deepfeatures/BlackOak", column_id=4,
                             text_file="/home/felix/SequentialPatternErrorDetection/address.txt"):
    matrix, size_list = read_all_deep_features(data_path, column_id, text_file)

    features = np.zeros((matrix.shape[0],matrix.shape[2]))

    for i in range(matrix.shape[0]):
        features[i] = matrix[i][size_list[i] - 1]

    return features


def deep_features_avg_state(data_path="/home/felix/SequentialPatternErrorDetection/deepfeatures/BlackOak", column_id=4,
                             text_file="/home/felix/SequentialPatternErrorDetection/address.txt"):
    matrix, size_list = read_all_deep_features(data_path, column_id, text_file)

    features = np.zeros((matrix.shape[0], matrix.shape[2]))

    for i in range(matrix.shape[0]):
        features[i] = np.mean(matrix[i][0:(size_list[i])], axis=0)

    return features


parser = OptionParser()
parser.add_option("-d", "--data", dest="data",
                  help="data folder", metavar="FOLDER")
parser.add_option("-c", "--columnid", dest="columnid",
                  help="column id")
parser.add_option("-o", "--output", dest="output",
                  help="output", metavar="FILE")

parser.add_option("-a", "--approach", dest="approach",
                  help="output")

(options, args) = parser.parse_args()

orig_file = options.data + "/column_" + str(options.columnid) + "/orig_input/column_" + str(options.columnid) + ".txt"

'''
orig_file = "/home/felix/SequentialPatternErrorDetection/address.txt"
options.approach = 1
options.data = "/home/felix/SequentialPatternErrorDetection/deepfeatures/BlackOak"
options.output = "/tmp/test.npz"
options.columnid = 4
'''


if options.approach == '1':
    features = deep_features_last_state(options.data, options.columnid, orig_file)
elif options.approach == '2':
    features = deep_features_avg_state(options.data, options.columnid, orig_file)
else:
    print "choose either approach 1 or 2"

np.savez_compressed(options.output, features)

'''
loaded = np.load(options.output)

loaded_features = loaded['arr_0']

assert np.all(np.equal(loaded_features,features))
'''
