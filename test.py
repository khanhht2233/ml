import numpy as np
import pandas as pd

a = np.array([1, 2 ,3, 4])
b = pd.Series(a, index=['a', 'b', 'c', 'D'])
print(b)
s = b[b > b.median()]
print(s)

