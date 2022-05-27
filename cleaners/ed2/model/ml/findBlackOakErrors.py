import numpy as np

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase

data = BlackOakDataSetUppercase()

for i in range(data.shape[1]):
    print str(data.clean_pd.columns[i]) + ": " + str(np.sum(data.matrix_is_error[:,i]))