from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
import numpy as np

#data = FlightHoloClean()
data = BlackOakDataSetUppercase()
#data = HospitalHoloClean()

relative = True

error_dist = np.zeros(data.shape[1])

errors_per_row = np.sum(data.matrix_is_error, axis=1)

for i in range(data.shape[0]):
    error_dist[errors_per_row[i]] += 1

if relative:
    error_dist /= data.shape[0]

print errors_per_row.shape


import matplotlib.pyplot as plt


fig, ax = plt.subplots()

index = np.arange(data.shape[1])
bar_width = 0.35

rects1 = plt.bar(index, error_dist, bar_width,
                 color='b',
                 label='Errors')

plt.xlabel('errors per row')
plt.ylabel('count')
plt.title('Error distribution')
plt.legend()

plt.tight_layout()
plt.show()

