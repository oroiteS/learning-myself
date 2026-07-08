import random

import matplotlib.pyplot as plt
import torch


def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # [num_examples, len(w)]
    y = X @ w + b  # [num_examples, 2]
    y += torch.normal(0, 0.01, y.shape)  # [num_examples, 2]
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print('features:', features[0], '\nlabel:', labels[0])

    plt.figure(figsize=(5, 3.5))
    plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)
    plt.show()

    batch_size = 10

    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break


if __name__ == "__main__":
    main()
