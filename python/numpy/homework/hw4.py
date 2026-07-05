
# %%
import numpy as np 

array = np.random.randint(0,10,size=(3,4))

print(array)
print(np.amax(array, axis=0))
print(np.amin(array, axis=1))
print(np.where(array % 2 ==1, -1, array))

# %%

