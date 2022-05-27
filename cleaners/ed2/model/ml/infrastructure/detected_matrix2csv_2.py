import numpy as np
from ml.datasets.HoloClean.HospitalNew import HospitalNew



data = HospitalNew()

columns = data.clean_pd.columns

print columns
print list(data.clean_pd.columns)


#detected = np.load("/home/felix/ExampleDrivenErrorDetection/model/ml/save_detected.npy")
detected = data.matrix_is_error

f = open("/tmp/dirty.csv", "w+")
f.write('ind,attr\n')
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if detected[row, col]:
            f.write(str(row + 1) + ',' + columns[col] + '\n')
f.close()

f = open("/tmp/clean.csv", "w+")
f.write('ind,attr\n')
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if not detected[row, col]:
            f.write(str(row + 1) + ',' + columns[col] + '\n')
f.close()
