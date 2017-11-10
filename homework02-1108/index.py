import glob
all_file_names = glob.glob('./data/names/*.txt')


# 在我们收集的18种语言的名字中，中文、日文、韩文等名字都已经转化为音译的字母。这样做是因为有些语言的名字并不能用普通的ASCII英文字符来表示，比如“Ślusàrski”，这些不一样的字母会增加神经网络的“困惑”，影响其训练效果。所以我们得首先把这些特别的字母转换成普通的ASCII字符（即26个英文字母）。
import unicodedata
import string

# 使用26个英文字母大小写再加上.,;这三个字符
# 建立字母表，并取其长度
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# 将Unicode字符串转换为纯ASCII
def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

# 然后再建立 readLines 方法，用于从文件中一行一行的将姓氏读取出来。
# 以18种语言为索引，将读取出的姓氏各自存储在名为 category_lines 的字典中。
# 构建category_lines字典，名字和每种语言对应的列表
category_lines = {}
all_categories = []

# 按行读取出名字并转换成纯ASCII
def readLines(filename):
	lines = open(filename).read().strip().split('\n')
	return [unicode_to_ascii(line) for line in lines]

for filename in all_file_names:
	# 取出每个文件的文件名（语言名）
	category = filename.split('/')[-1].split('.')[0]
	# 将语言名加入到all_categories列表
	all_categories.append(category)
	# 取出所有的姓氏lines
	lines = readLines(filename)
	# 将所有姓氏以语言为索引，加入到字典中
	category_lines[category] = lines

n_categories = len(all_categories)


# 现在开始正式的训练
# 首先导入程序所需要的程序包
#PyTorch用的包
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable


#绘图、计算用的程序包
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np


# 下面我们再编写一个方法用于快速地获得一个训练实例（即一个名字以及它所属的语言）：
# 其中 line_index 中保存的是选择的姓氏中的字母的索引，这个需要你去实现。
import random

def random_training_pair():   
	# 随机选择一种语言
	category = random.choice(all_categories)
	# 从语言中随机选择一个姓氏
	line = random.choice(category_lines[category])
	# 我们将姓氏和语言都转化为索引
	category_index = all_categories.index(category)

	# line_index = []
	# 你需要把 line 中字母的索引加入到line_index 中
	# Todo:
	line_index = [all_letters.index(letter) for letter in line]

	return category, line, category_index, line_index

# 现在是建立 LSTM 模型的时候了。
class LSTMNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers=1):
		super(LSTMNetwork, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size

		# LSTM的构造如下：
		# 一个embedding层，将输入的任意一个单词（list）映射为一个向量（向量的维度与隐含层有关系？）
		self.embedding = nn.Embedding(input_size,hidden_size)
		# 然后是一个LSTM隐含层，共有hidden_size个LSTM神经元，并且它可以根据n_layers设置层数
		self.lstm = nn.LSTM(hidden_size,hidden_size,n_layers)
		# 接着是一个全链接层，外接一个softmax输出
		self.fc = nn.Linear(hidden_size,output_size)
		self.logsoftmax = nn.LogSoftmax()

	def forward(self, input, hidden=None):
		#首先根据输入input，进行词向量嵌入
		embedded = self.embedding(input)

		# 这里需要注意！
		# PyTorch设计的LSTM层有一个特别别扭的地方是，输入张量的第一个维度需要是时间步，
		# 第二个维度才是batch_size，所以需要对embedded变形
		# 因为此次没有采用batch，所以batch_size为1
		# 变形的维度应该是（input_list_size, batch_size, hidden_size）
		embedded = embedded.view(input.data.size()[0],1,self.hidden_size)

		# 调用PyTorch自带的LSTM层函数，注意有两个输入，一个是输入层的输入，另一个是隐含层自身的输入
		# 输出output是所有步的隐含神经元的输出结果，hidden是隐含层在最后一个时间步的状态。
		# 注意hidden是一个tuple，包含了最后时间步的隐含层神经元的输出，以及每一个隐含层神经元的cell的状态

		output, hidden = self.lstm(embedded, hidden)

		#我们要把最后一个时间步的隐含神经元输出结果拿出来，送给全连接层
		output = output[-1,...]

		#全链接层
		out = self.fc(output)
		# softmax
		out = self.logsoftmax(out)
		return out

	def initHidden(self):
		# 对隐单元的初始化
		# 对引单元输出的初始化，全0.
		# 注意hidden和cell的维度都是layers,batch_size,hidden_size
		hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
		# 对隐单元内部的状态cell的初始化，全0
		cell = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
		return (hidden, cell)


# 开始训练网络
import time
import math

# 开始训练LSTM网络
n_epochs = 100000

# 构造一个LSTM网络的实例
lstm = LSTMNetwork(n_letters, 10, n_categories, 2)

#定义损失函数
cost = torch.nn.NLLLoss()

#定义优化器,
optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)
records = []

# 用于计算训练时间的函数
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

# 统计所有名字的个数
all_line_num = 0
for key in category_lines:
    all_line_num += len(category_lines[key])
print(all_line_num)


def category_from_output(output):
	# 1 代表在‘列’间找到最大
	# top_n 是具体的值
	# top_i 是位置索引
	# 注意这里 top_n 和 top_i 都是1x1的张量
	# output.data 取出张量数据
	top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
	# 从张量中取出索引值
	category_i = top_i[0][0]
	# 返回语言类别名和位置索引
	return all_categories[category_i], category_i


# 开始训练，一共5个epoch，否则容易过拟合
for epoch in range(5):
    losses = []
    #每次随机选择数据进行训练，每个 EPOCH 训练“所有名字个数”次。
    for i in range(all_line_num):
        category, line, y, x = random_training_pair()
        x = Variable(torch.LongTensor(x))
        y = Variable(torch.LongTensor(np.array([y])))
        optimizer.zero_grad()
        
        # Step1:初始化LSTM隐含层单元的状态
        hidden = lstm.initHidden()
        
        # Step2:让LSTM开始做运算，注意，不需要手工编写对时间步的循环，而是直接交给PyTorch的LSTM层。
        # 它自动会根据数据的维度计算若干时间步
        output = lstm(x,hidden)
        
        # Step3:计算损失
        loss = cost(output,y)
        
        losses.append(loss.data.numpy()[0])
        
        #反向传播
        loss.backward()
        optimizer.step()
        
        #每隔3000步，跑一次校验集，并打印结果
        if i % 3000 == 0:
            # 判断模型的预测是否正确
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 计算训练进度
            training_process = (all_line_num * epoch + i) / (all_line_num * 5) * 100
            training_process = '%.2f' % training_process
            print('第{}轮，训练损失：{:.2f}，训练进度：{}%，（{}），名字：{}，预测国家：{}，正确？{}'\
                .format(epoch, np.mean(losses), float(training_process), time_since(start), line, guess, correct))
            records.append([np.mean(losses)])





