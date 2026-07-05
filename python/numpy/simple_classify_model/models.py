# %%
# 生成数据集 
import make_moons
import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
def initialize_parameters():
    W1 = np.random.randn(2,4)
    b1 = np.random.randn(1,4)
    W2 = np.random.randn(4,1)
    b2 = np.random.randn(1,1)
    return W1,b1,W2,b2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def calculate_loss(x, y):
    n = y.shape[0]
    x=np.clip(x,1e-15,1-1e-15)
    loss = -(1/n)*np.sum(y*np.log(x)+(1-y)*np.log(1-x))
    return loss

def forward(x, params):
    W1,W2,b1,b2=params['W1'],params['W2'],params['b1'],params['b2']
    z1=x@W1+b1
    a1=np.tanh(z1)
    z2=a1@W2+b2
    a2=sigmoid(z2)
    
    cache = {}
    cache['A1'] = a1
    cache['A2'] = a2

    return cache

def backward(x, y, cache, params, learning_rate):
    n = y.shape[0]
    A2, A1 = cache['A2'], cache['A1']
    W1,W2,b1,b2=params['W1'],params['W2'],params['b1'],params['b2']
    A2 = np.clip(A2,1e-15,1-1e-15)
    dA2 = -(1/n)*(y/A2-(1-y)/(1-A2))
    dZ2 = dA2 * A2 * (1-A2)
    dW2 = A1.T @ dZ2
    dB2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (1-A1**2)
    dW1 = x.T @ dZ1
    dB1 = np.sum(dZ1, axis=0, keepdims=True)
    
    params['W2'] = W2 - learning_rate * dW2
    params['W1'] = W1 - learning_rate * dW1
    params['b1'] = b1 - learning_rate * dB1
    params['b2'] = b2 - learning_rate * dB2
    return params

def train(learning_rate=0.001, epoch=100):
    X,Y = make_moons.generate_moons(200,0.1)

    W1, b1, W2, b2 = initialize_parameters()
    params = {}
    params['W1'] = W1
    params['W2'] = W2
    params['b1'] = b1
    params['b2'] = b2
    
    cache = {}
    loss_history = []

    for i in range(epoch):
        cache = forward(X, params)
        loss = calculate_loss(cache['A2'], Y)
        loss_history.append(loss)
        params = backward(X, Y, cache, params, learning_rate)
        print(f"epoch{i}:{loss}")
    
    return loss_history

def show(loss_history):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train_Loss")
    plt.show()

# %%
def main():
    loss_his = train()
    show(loss_history=loss_his)

main()
# %%

