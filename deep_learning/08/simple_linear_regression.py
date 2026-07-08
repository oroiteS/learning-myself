import torch
from data import data_iter, synthetic_data


def linreg(X, w, b):
    """线性回归模型"""
    return X @ w + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, learning_rate, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad / batch_size
            param.grad.zero_()


true_w = torch.tensor([2, -3.4])  # [1,2]
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  # features:[]

print('features:', features[0], '\nlabel:', labels[0])

batch_size = 10
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
