import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data import get_fashion_mnist_labels, load_data_fashion_mnist


class MLP():
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    def forward(self, X):
        X = X.reshape((-1, self.W1.shape[0]))
        H = relu(X @ self.W1 + self.b1)
        return (H @ self.W2 + self.b2)

    def __call__(self, X):
        return self.forward(X)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def train(model, num_epochs, num_input, train_iter, test_iter, loss_fn, optimizer):
    for epoch in range(num_epochs):
        total_loss, total_acc, total_num = 0., 0., 0
        for X, y in train_iter:
            X = X.reshape(-1, num_input)
            y_hat = model(X)
            l = loss_fn(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_loss += l.item() * X.shape[0]
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            total_num += X.shape[0]

        train_loss = total_loss / total_num
        train_acc = total_acc / total_num

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
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(model(X).argmax(axis=1))
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
