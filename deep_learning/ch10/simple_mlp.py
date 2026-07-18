import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_fashion_mnist_labels, load_data_fashion_mnist


class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(num_inputs, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_outputs)
        )

    def forward(self, X):
        return self.net(X)


def train(model, num_epochs, num_input, train_iter, test_iter, loss_fn, optimizer):
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc, total_num = 0., 0., 0
        for X, y in train_iter:
            X = X.reshape(-1, num_input)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.shape[0]
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            total_num += X.shape[0]

        train_loss = total_loss / total_num
        train_acc = total_acc / total_num

        model.eval()
        test_acc = 0.
        with torch.no_grad():
            for X, y in test_iter:
                X = X.reshape(-1, 784)
                y_hat = model(X)
                test_acc += (y_hat.argmax(dim=1) == y).sum().item()
        test_acc /= len(test_iter.dataset)
        print(
            f'Epoch {epoch+1}: train loss {train_loss:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}'
        )


def predict(model, test_iter, n=6):
    X, y = next(iter(test_iter))
    model.eval()
    with torch.no_grad():
        X_flat = X.reshape(-1, 784)  # (batch, 784)
        y_hat = model(X_flat)  # 输出 logits
        preds = y_hat.argmax(dim=1)
    # 获取标签文本
    trues = get_fashion_mnist_labels(y)
    pred_labels = get_fashion_mnist_labels(preds)

    # 准备绘图
    titles = [true + '\n' + pred for true, pred in zip(trues, pred_labels)]
    # 取前 n 张图像：X 形状 (batch, 1, 28, 28)，去掉通道维并转为 numpy
    images = X[:n, 0, :, :].numpy()  # 形状 (n, 28, 28)

    # 创建子图
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i], fontsize=10)
        axes[i].axis('off')
    plt.show()


def main():
    config = {
        'batch_size': 256,
        'dataloader_workers': 4,
        'num_inputs': 784,
        'num_outputs': 10,
        'num_hiddens': 256,
        'num_epoch': 10,
        'lr': 0.1
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model = MLP(config['num_inputs'], config['num_hiddens'], config['num_outputs'])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    train_iter, test_iter = load_data_fashion_mnist(
        config['batch_size'], config['dataloader_workers'], script_dir
    )
    train(model, config['num_epoch'], config['num_inputs'], train_iter, test_iter, loss, optimizer)
    predict(model, test_iter)


if __name__ == '__main__':
    main()
