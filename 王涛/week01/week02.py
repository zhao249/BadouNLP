import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，第几个数大，就为相应的类别

"""
# 5*10 
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(5, 10) # 线性层
        self.linear2 = nn.Linear(10, 5) # 线性层
        # 交叉熵损失函数
        self.loss= nn.functional.cross_entropy
    def forward(self, x,y=None):
        x=self.linear(x)  # (batch_size, input_size) -> (batch_size, 10)
        y_pred = self.linear2(x)  # (batch_size, 10) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    correct, wrong = 0, 0
    x,y= build_dataset(test_sample_num)
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 训练模型
def train():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    log = []
    # 建立模型
    model = TorchModel(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # 训练集
    train_x,train_y = build_dataset(train_sample)
    # 训练
    for i in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx in range(train_sample//batch_size):
            x=train_x[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y=train_y[batch_idx*batch_size:(batch_idx+1)*batch_size]
            loss=model.forward(x,y) 
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
        acc=evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
        # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__  == "__main__":
    # train()
    x,y=build_dataset(1000)
    print(y.shape)

