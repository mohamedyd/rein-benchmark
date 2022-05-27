import numpy as np

def top_k_hybrid(a, k):
    b = np.argpartition(a, k)[:k]
    return a[b[np.argsort(a[b])]]

a = np.array([1,1,-1,2,3,4,-5])


#print top_k_hybrid(a,3)

ids = np.argsort(a)

print a[ids]
