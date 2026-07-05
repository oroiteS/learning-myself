
# %%
import numpy as np 

array = np.random.randint(0,20,(5,5))
print(array)
print(array[array > 10])
print(np.where(array>10,0,array))

# %%

