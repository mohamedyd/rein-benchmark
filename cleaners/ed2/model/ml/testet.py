import numpy as np

arraym = np.array([0.1, 0.4, 0.2])
sorted_ids = np.argsort(arraym)

print sorted_ids
print arraym[sorted_ids]