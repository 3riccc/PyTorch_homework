# 导入需要的包
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import numpy as np


# 这次的数据仍然是18个文本文件，每个文件以“国家名字”命名，文件中存储了大量这个国家的姓氏。
# 在读取这些数据前，为了简化神经网络的输入参数规模，我们把各国各语言人名都转化成用26个英文字母来表示，下面就是转换的方法。
import glob
import unicodedata
import string

# all_letters 即课支持打印的字符+标点符号
all_letters = string.ascii_letters + " .,;'-"
# Plus EOS marker
n_letters = len(all_letters) + 1 
EOS = n_letters - 1

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )



# 准备好处理数据的方法，下面就可以放心的读取数据了。
# 我们建立一个列表 all_categories 用于存储所有的国家名字。
# 建立一个字典 category_lines，以读取的国名作为字典的索引，国名下存储对应国别的名字。
# 和上一周作业一样
# 按行读取出文件中的名字，并返回包含所有名字的列表
def read_lines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# category_lines是一个字典
# 其中索引是国家名字，内容是从文件读取出的这个国家的所有名字
category_lines = {}
# all_categories是一个列表
# 其中包含了所有的国家名字
all_categories = []
# 循环所有文件
for filename in glob.glob('./names/*.txt'):
    # 从文件名中切割出国家名字
    category = filename.split('/')[-1].split('.')[0]
    # 将国家名字添加到列表中
    all_categories.append(category)
    # 读取对应国别文件中所有的名字
    lines = read_lines(filename)
    # 将所有名字存储在字典中对应的国别下
    category_lines[category] = lines

# 共有的国别数
n_categories = len(all_categories)



# 再统计下手头共有多少条训练数据
all_line_num = 0
for key in category_lines:
    all_line_num += len(category_lines[key])


import random
def random_training_pair():
    # 随机选择一个国别名
    category = random.choice(all_categories)
    # 读取这个国别名下的所有人名
    line = random.choice(category_lines[category])
    return category, line


# 将名字所属的国家名转化为“独热向量”
def make_category_input(category):
    li = all_categories.index(category)
    return  li


# 对于训练过程中的每一步，或者说对于训练数据中每个名字的每个字符来说，神经网络的输入是 (category, current letter, hidden state)，输出是 (next letter, next hidden state)。
# 与在课程中讲的一样，神经网络还是依据“当前的字符”预测“下一个字符”。比如对于“Kasparov”这个名字，创建的（input, target）数据对是 ("K", "a"), ("a", "s"), ("s", "p"), ("p", "a"), ("a", "r"), ("r", "o"), ("o", "v"), ("v", "EOS")。

def make_chars_input(nameStr):
    name_char_list = list(map(lambda x: all_letters.find(x), nameStr))
    return name_char_list


def make_target(nameStr):
    target_char_list = list(map(lambda x: all_letters.find(x), nameStr[1:]))
    target_char_list.append(n_letters - 1)# EOS
    return target_char_list

def random_training_set():
    # 随机选择数据集
    category, line = random_training_pair()
    #print(category, line)
    # 转化成对应 Tensor
    category_input = make_category_input(category)
    line_input = make_chars_input(line)
    #category_name_input = make_category_name_input(category, line)
    line_target = make_target(line)
    return category_input, line_input, line_target



# 搭建神经网络
# 这次使用的 LSTM 神经网络整体结构上与课上讲的生成音乐的模型非常相似，不过有一点请注意一下。
# 我们要把国别和国别对应的姓氏一同输入到神经网络中，这样 LSTM 模型才能分别学习到每个国家姓氏的特色，从而生成不同国家不同特色的姓氏。
# 那国别数据与姓氏数据应该如何拼接哪？应该在嵌入前拼接，还是在嵌入后再进行拼接哪？嵌入后的维度与 hidden_size 有怎样的关系哪？
# 一个手动实现的LSTM模型，

class LSTMNetwork(nn.Module):
    def __init__(self, name_size, hidden_size, num_layers = 1):
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
       
        # 进行嵌入
        self.embedding = nn.Embedding(name_size,hidden_size)
        
        # 隐含层内部的相互链接
        self.lstm = nn.LSTM(hidden_size,hidden_size,num_layers)
        
        # 输出层
        self.fc = nn.Linear(hidden_size,name_size)
        self.logsoftmax = nn.LogSoftmax()
        

    def forward(self,name_variable, hidden):
        
        # 先分别进行embedding层的计算
        embedded = self.embedding(name_variable)
        embedded = embedded.view(name_variable.data.size()[0],1,self.hidden_size)
        
        # 从输入到隐含层的计算
        output, hidden = self.lstm(embedded, hidden)
        
        # output的尺寸：batch_size, len_seq, hidden_size
        output = output[:,-1,:]

        # 全连接层
        output = self.fc(output)
        # output的尺寸：batch_size, output_size
        # softmax函数
        output = self.logsoftmax(output)
        

        return output, hidden
 
    def initHidden(self):
        # 对隐含单元的初始化
        # 注意尺寸是： layer_size, batch_size, hidden_size
        # 对隐单元的初始化
        # 对引单元输出的初始化，全0.
        # 注意hidden和cell的维度都是layers,batch_size,hidden_size
        hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        # 对隐单元内部的状态cell的初始化，全0
        cell = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        return (hidden, cell)



HIDDEN_SIZE = 10
num_epoch = 3
learning_rate = 0.001

# 实例化模型
lstm = LSTMNetwork(n_letters,128,1)
# 定义损失函数与优化方法
optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)
cost = torch.nn.NLLLoss() #交叉熵损失函数

# 定义训练函数，在这个函数里，我们可以随机选择一条训练数据，遍历每个字符进行训练
def train_LSTM():
    # 初始化 隐藏层、梯度清零、损失清零
    hidden = lstm.initHidden()
    optimizer.zero_grad()
    loss = 0
    
    
    # 随机选取一条训练数据
    category_input, line_input, line_target = random_training_set()
    line_target = [line_input[0]] + line_target
    line_input = [category_input] + line_input
    # 处理国别数据
    # category_variable 
    
    # 循环字符
    for t in range(len(line_input)):
        # 姓氏
        x = Variable(torch.LongTensor([line_input[t]]).unsqueeze(0))
        # 目标
        y = Variable(torch.LongTensor(np.array([line_target[t]])))
        # 传入模型
        output,hidden = lstm(x,hidden)
        output = output.view(1,n_letters)
        # 累加损失
        loss += cost(output, y)
        
    
    # 计算平均损失
    loss = 1.0 * loss / len(line_input) 
    # 反向传播、更新梯度
    loss.backward()
    optimizer.step()
    
    return loss
import time
import math

def time_since(t):
    now = time.time()
    s = now - t
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
start = time.time()

num_epoch = 1
records = []
# 开始训练循环
for epoch in range(num_epoch):
    train_loss = 0
    # 按所有数据的行数随机循环
    for i in range(all_line_num):
    # for i in range(10):
        loss = train_LSTM()
        train_loss += loss
        
        #每隔3000步，跑一次校验集，并打印结果
        if i % 1000 == 0:
        	print(loss)
        	# training_process = (all_line_num * epoch + i)/(all_line_num*num_epoch)*100
         #    training_process='%.2f'%training_process
         #    print('第{}轮，训练损失：{:.2f}，训练进度：{}%，（{}）'\
         #    	.format(epoch, train_loss.data.numpy()[0] / i, float(training_process), time_since(start)))
         #    records.append([train_loss.data.numpy()[0] / i])



# 通过指定国别名 category
# 以及开始字符 start_char
# 还有混乱度 temperature 来生成一个名字
def generate_one(category ,temperature=0.2):
    # 初始化输入数据，国别 以及 输入的第一个字符
    # 国别
    # 因为训练时就做了拼接，所以国别作为第一个字符统一处理
    # 第一个字符
    top_i = make_category_input(category)
    char = Variable(torch.LongTensor([top_i]).unsqueeze(0))
    # 初始化隐藏层
    hidden = lstm.initHidden()
    name = [top_i]
    # output_str = start_char
    # 因为国别统一处理，也不需要第一个字符，第一个字符应该会自动生成
    i = 0
    # for i in range(1,10):

    while top_i != EOS:
    	output,hidden = lstm(char,hidden)
    	output_dist = output.data.view(-1).div(temperature).exp()
    	top_i = torch.multinomial(output_dist, 1)[0]
    	name.append(top_i)
    	char = Variable(torch.LongTensor([top_i]).unsqueeze(0))
    # 将数字序列转化为字母序列
    name_str = ''.join([all_letters[name[i]] for i in range(1,len(name)-1)])
    print(name_str)
    return name_str



generate_one("Chinese")
generate_one("Chinese")
generate_one("Chinese")
generate_one("Chinese")
generate_one("Chinese")
generate_one("Chinese")
generate_one("Chinese")
