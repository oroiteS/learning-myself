
# %%
import numpy as np 

a, b = np.array([1, 2, 3]), np.array([4, 5, 6])

print(np.concatenate([a, b]))
print(np.array([a,b]))
print(np.concatenate([a, b]).reshape(2, 3))

# %%

