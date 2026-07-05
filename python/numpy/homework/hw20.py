
# %%
import numpy as np 

sales = np.array([20, 25, 22, 30, 28])
costs = np.array([15, 18, 16, 22, 20])

gainings = sales - costs
print(gainings)

print(np.mean(gainings))
print(np.std(gainings))

print(np.argmax(gainings)+1)
# %%

