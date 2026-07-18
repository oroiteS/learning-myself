import torch
import torch.nn as nn
from data import data_iter, synthetic_data
from torch.utils import data


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 pytorch 数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
batch_size = 10
num_epoch = 4

features, labels = synthetic_data(true_w, true_b, 1000)  # 生成数据集
data_iter = load_array((features, labels), batch_size)  # 读取数据集
net = nn.Sequential(nn.Linear(2, 1))  # 网络结构
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
for epoch in range(num_epoch):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
