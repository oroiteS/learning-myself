
# %%
import numpy as np
def generate_moons(n_samples=200, noise=0.1):
    n_out = n_samples // 2
    # 上半月
    theta = np.linspace(0, np.pi, n_out)
    X1 = np.column_stack([np.cos(theta), np.sin(theta)]) + noise * np.random.randn(n_out, 2)
    # 下半月
    X2 = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)]) + noise * np.random.randn(n_out, 2)
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_out), np.ones(n_out)]).reshape(-1, 1)
    # 打乱
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]

X,Y=generate_moons(200,0.1)
print(X.shape)
print(X.ndim)
print(Y.shape)
print(Y.ndim)
# %%

