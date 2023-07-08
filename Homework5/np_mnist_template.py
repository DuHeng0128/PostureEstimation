# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    return np.maximum(0,x)

def relu_prime(x):
    '''
    relu函数的导数
    '''
    return np.where(x > 0, 1, 0)

#输出层激活函数
def softmax(x):
    '''
    softmax函数, 防止除0
    '''
    x=x-np.max(x,axis=-1,keepdims=True)
    return np.exp(x)/ np.sum(np.exp(x),axis=-1,keepdims=True)

def f_prime(x):
    '''
    softmax函数的导数
    '''
    return softmax(x)*(1-softmax(x))


# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return np.sum(y_true*np.log(y_pred+1e-8),axis=-1)

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return y_pred-y_true



# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    np.random.seed(seed=1000)
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape,)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        '''
        self.W1 = init_weights((input_size, hidden_size)) # 输入层到隐藏层的权重矩阵
        self.b1 = init_weights((hidden_size,)) # 输入层到隐藏层的偏置
        self.W2 = init_weights((hidden_size, output_size)) # 隐藏层到输出层的权重矩阵
        self.b2 = init_weights((output_size,)) # 隐藏层到输出层的偏置
        self.lr = lr # 学习率

    def forward(self, X):
        z1  = np.matmul(X, self.W1) + self.b1 # z^{L-1}
        a1  = relu(z1) # a^{L-1}
        z2  = np.matmul(a1, self.W2) + self.b2 # z^{L}
        a2 = softmax(z2) # a^{L}
        return a1,a2

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        self.grads_W2 = np.zeros_like(self.W2)
        self.grads_b2 = np.zeros_like(self.b2)
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)
        # 前向传播
        a1,a2 = self.forward(x_batch)
        # 计算损失和准确率
        loss = - (y_batch*np.log(a2+1e-8) + (1-y_batch)*np.log(1-a2+1e-8)) 
        accuracy = np.where(np.argmax(y_batch, axis=1) == np.argmax(a2, axis=1), 1, 0).reshape(-1, 1)
        delta_L = - (y_batch * (1-a2) - (1-y_batch) * a2 )
        delta_l = np.matmul(delta_L,self.W2.T) * relu_prime(a1)
        
        loss = np.mean(loss)
        accuracy = np.mean(accuracy)

        # 反向传播
        self.grads_W2 += np.mean(np.expand_dims(a1, axis=-1) * delta_L.reshape(delta_L.shape[0], 1, delta_L.shape[1]), axis=0)
        self.grads_b2 += np.mean(delta_L,axis=0)
        self.grads_W1 += np.mean(np.expand_dims(x_batch,axis=-1) * delta_l.reshape(delta_l.shape[0], 1, delta_l.shape[1]), axis=0)
        self.grads_b1 += np.mean(delta_l,axis=0)
        

        # 更新权重
        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1

        return loss,accuracy
    
    def predict(self, X):
        a1,a2=self.forward(X)
        return np.argmax(a2, axis=1)

if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.02)
    
    for epoch in range(10):
        losses = 0
        accuracies = 0
        num = len(X_train)//64

        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i:i+64]
            y_batch = y_train[i:i+64]
            loss,accuracy = net.step(x_batch,y_batch)
            losses += loss
            accuracies += accuracy
        losses /= num   
        accuracies /= num
        print('Epoch {}/{}: train_loss={:.4f}, train_accuracy={:.4f}'.format(epoch+1, 10, losses, accuracies)) 
    
    bool_array = (net.predict(X_test) == np.argmax(y_test, axis=1))   
    test_accuracy = bool_array.astype(int).mean()
    print("Test Accuracy: %.4f" % test_accuracy)    