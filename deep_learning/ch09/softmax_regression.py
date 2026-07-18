import os

import matplotlib.pyplot as plt
import torch
from data import get_fashion_mnist_labels, load_data_fashion_mnist


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据（纯脚本环境适用）"""
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale='linear',
        yscale='linear',
        fmts=('-', 'm--', 'g-.', 'r:'),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5)
    ):
        if legend is None:
            legend = []
        plt.ion()  # 开启交互模式，允许动态更新
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        # 保存绘图参数
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.legend = legend
        self.fmts = fmts
        self.X, self.Y = None, None

    def _config_axes(self):
        """设置坐标轴属性"""
        ax = self.axes[0]
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.xscale:
            ax.set_xscale(self.xscale)
        if self.yscale:
            ax.set_yscale(self.yscale)
        if self.legend:
            ax.legend(self.legend)

    def add(self, x, y):
        """添加数据点并刷新图形"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        # 清除当前轴并重绘
        ax = self.axes[0]
        ax.cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            ax.plot(x_vals, y_vals, fmt)
        self._config_axes()
        # 刷新画布
        self.fig.canvas.draw()
        plt.pause(0.001)  # 短暂暂停以更新窗口
        """添加数据点并刷新图形"""
        if not hasattr(y, "__len__"):
            y = [y]


class SoftmaxRegression():
    def __init__(self, num_input, num_output):
        self.W = torch.normal(0, 0.01, size=(num_input, num_output), requires_grad=True)
        self.b = torch.zeros(num_output, requires_grad=True)

    def net(self, X):
        return softmax(X.reshape((-1, self.W.shape[0])) @ self.W + self.b)

    def parameters(self):
        return [self.W, self.b]


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def sgd(model, lr, batch_size):
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad / batch_size
            p.grad.zero_()


def evaluate_accuracy(model, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = model.net(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]


def train_epoch(model, train_iter, loss, updater, lr=0.01):
    """训练模型一个迭代周期 """
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = model.net(X)
        l = loss(y_hat, y)
        l.sum().backward()
        updater(model, lr, X.shape[0])
        with torch.no_grad():
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(model, train_iter, test_iter, loss, num_epochs, lr, updater):
    animator = Animator(
        xlabel='epoch',
        xlim=[1, num_epochs],
        ylim=[0.3, 0.9],
        legend=['train loss', 'train acc', 'test acc']
    )
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_iter, loss, updater, lr)
        test_acc = evaluate_accuracy(model, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    print(f'训练损失: {train_loss:.3f}, 训练精度: {train_acc:.3f}, 测试精度: {test_acc:.3f}')


def predict(model, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(model.net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    images = X[0:n].reshape((n, 28, 28))
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axes = [axes]
    for i, (img, title) in enumerate(zip(images, titles[0:n])):
        axes[i].imshow(img.numpy(), cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.show()


def main():
    config = {
        'num_inputs': 784,
        'num_outputs': 10,
        'lr': 0.1,
        'batch_size': 256,
        'num_epochs': 10,
        'dataloader_workers': 4,
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载数据
    train_iter, test_iter = load_data_fashion_mnist(
        config['batch_size'], config['dataloader_workers'], script_dir
    )
    # 2. 实例化模型（W 和 b 在内部初始化，再也不用传了！）
    model = SoftmaxRegression(config['num_inputs'], config['num_outputs'])

    # 3. 训练（只传 model，不传 W 和 b）
    train(model, train_iter, test_iter, cross_entropy, config['num_epochs'], config['lr'], sgd)

    # 4. 预测
    predict(model, test_iter)


if __name__ == "__main__":
    main()
