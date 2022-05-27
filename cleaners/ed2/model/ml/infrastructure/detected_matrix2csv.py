import numpy as np
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean



data = HospitalHoloClean()

columns = ["ProviderNumber",
           "HospitalName",
           "Address1",
           "City",
           "State",
           "ZipCode",
           "CountyName",
           "PhoneNumber",
           "HospitalType",
           "HospitalOwner",
           "EmergencyService",
           "Condition",
           "MeasureCode",
           "MeasureName",
           "Score",
           "Sample",
           "Stateavg"]

print columns
print list(data.clean_pd.columns)


#detected = np.load("/home/felix/ExampleDrivenErrorDetection/model/ml/save_detected.npy")
detected = data.matrix_is_error

f = open("/tmp/dirty.csv", "w+")
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if detected[row, col]:
            f.write(str(row + 1) + ',' + columns[col] + '\n')
f.close()

f = open("/tmp/clean.csv", "w+")
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if not detected[row, col]:
            f.write(str(row + 1) + ',' + columns[col] + '\n')
for row in range(data.shape[0]):
    for colum_name in ['Address2', 'Address3']:
        f.write(str(row + 1) + ',' + colum_name + '\n')

f.close()
