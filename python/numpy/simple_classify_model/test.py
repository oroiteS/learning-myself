# %%
import matplotlib.pyplot as plt

import numpy as np


# 复制你的函数（稍加修改以便观察原始无噪声形状）
def generate_moons_clean(n_samples=200):
    n_out = n_samples // 2
    theta = np.linspace(0, np.pi, n_out)
    # 上半月（无噪声）
    X1 = np.column_stack([np.cos(theta), np.sin(theta)])
    # 下半月（无噪声）
    X2 = np.column_stack([1 - np.cos(theta), 0.5 - np.sin(theta)])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_out), np.ones(n_out)])
    return X, y


# 生成数据
X, y = generate_moons_clean(n_samples=200)

# 绘图
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('双月形状（无噪声）\n红色=下半月(标签1), 蓝色=上半月(标签0)')
plt.axis('equal')
plt.grid(True)
plt.show()
# %%
