
# %%
import numpy as np 

arr = np.random.randint(1,100,(12))
print(arr)
arr = arr.reshape(3,4)
print(arr)
print(np.sum(arr,axis=0))  # 每列的和
print(np.sum(arr,axis=1))  # 每行的和
print(arr.reshape(1,12))

# %%

