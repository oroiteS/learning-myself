import torch

x = torch.arange(24).reshape(2, 3, 4)
print(x)
# print(x.t())
print(x.T)
print(x.mT)

x = torch.arange(40).reshape(2, 5, 4)
print(x)
y = x.sum(dim=[0, 1])
print(y.shape)
print(y)

y = x.sum(axis=(0, 1, 2))
print(y)
