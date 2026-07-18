# %%
import numpy as np

tems = np.array([28, 30, 29, 31, 32, 30, 29])
print(tems.max())
print(tems.min())
print(np.max(tems))
print(np.min(tems))
print(np.average(tems))

day_n = 0
for tem in tems:
    if tem > 30:
        day_n += 1

print(day_n)

# %%
print(len(tems[tems > 30]))
print(np.sum(np.where(tems > 30, 1, 0)))
print(np.count_nonzero(tems > 30))
# %%
