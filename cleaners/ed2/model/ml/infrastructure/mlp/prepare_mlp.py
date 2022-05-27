import numpy as np
from sklearn.preprocessing import normalize
import os

def run_command(command, run):
    print command
    if run:
        os.system(command)


data = np.load("storage/data.npy")
names = np.load("storage/names.npy")


hidden_units = 1024
learning_rate = 0.01


columns = []

for i in range(len(names)):
    if "_string_length" in names[i]:
        columns.append(names[i][0:-14])


for c in range(len(columns)):
    y_ids = []
    x_ids = []
    for i in range(len(names)):
        if columns[c] in names[i]:
            y_ids.append(i)
        else:
            x_ids.append(i)

    X = data[:, x_ids]
    Y = data[:, y_ids]

    print X.shape
    print Y.shape

    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)

    X_norm = normalize(X)
    Y_norm = normalize(Y)

    #np.save("/home/felix/SequentialPatternErrorDetection/model/ml/data/X_small.npy", X_norm[0:100,:])
    #np.save("/home/felix/SequentialPatternErrorDetection/model/ml/data/Y_small.npy", Y_norm[0:100,:])

    np.save("X.npy", X_norm)
    np.save("Y.npy", Y_norm)

    command = 'th train_mlp.lua' + \
              ' -input_units ' + str(len(x_ids)) + \
              ' -hidden_units ' + str(hidden_units) + \
              ' -output_units ' + str(len(y_ids)) + \
              ' -size ' + str(X.shape[0]) + \
              ' -output ' + 'loss/column' + str(c) + '.npy' + \
              ' -learning_rate ' + str(learning_rate) + \
              '\n\n'
    run_command(command, True)