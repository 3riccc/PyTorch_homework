# 用到的包
#from __future__ import unicode_literals, print_function, division
# 进行系统操作，如io、正则表达式的包
from io import open
import unicodedata
import string
import re
import random
#import time
#import math

#Pytorch必备的包
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.utils.data as DataSet


# 绘图所用的包
import matplotlib.pyplot as plt
import numpy as np


# 判断本机是否有支持的GPU
use_cuda = torch.cuda.is_available()

# 读取平行语料库
# 这是人民日报语料库
lines = open('data/chinese.txt', encoding = 'utf-8')
chinese = lines.read().strip().split('\n')
lines = open('data/english.txt', encoding = 'utf-8')
english = lines.read().strip().split('\n')


# 定义两个特殊符号，分别对应句子头和句子尾
SOS_token = 0
EOS_token = 1

# 定义一个语言类，方便进行自动的建立、词频的统计等
# 在这个对象中，最重要的是两个字典：word2index，index2word
# 故名思议，第一个字典是将word映射到索引，第二个是将索引映射到word
class Lang:
	def __init__(self,name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS

	def addSentence(self, sentence):
		# 在语言中添加一个新句子，句子是用空格隔开的一组单词
		# 将单词切分出来，并分别进行处理
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self,word):
		# 插入一个单词，如果单词已经在字典中，则更新字典中对应单词的频率
		# 同时建立反向索引，可以从单词编号找到单词
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
# 将unicode编码转变为ascii编码
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

# 把输入的英文字符串转成小写
def normalizeEngString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

# 对输入的单词对做过滤，保证每句话的单词数不能超过MAX_LENGTH
def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and \
		len(p[1].split(' ')) < MAX_LENGTH

# 输入一个句子，输出一个单词对应的编码序列
def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

# 和上面的函数功能类似，不同在于输出的序列等长＝MAX_LENGTH
def indexFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	for i in range(MAX_LENGTH - len(indexes)):
		indexes.append(EOS_token)
	return(indexes)


# 从一个词对到下标
def indexFromPair(pair):
	input_variable = indexFromSentence(input_lang, pair[0])
	target_variable = indexFromSentence(output_lang, pair[1])
	return (input_variable, target_variable)

# 从一个列表到句子
def SentenceFromList(lang, lst):
	result = [lang.index2word[i] for i in lst if i != EOS_token]
	if lang.name == 'Chinese':
		result = ' '.join(result)
	else:
		result = ' '.join(result)
	return(result)

# 计算准确度的函数
def rightness(predictions, labels):
	# print("predictions")
	# print(predictions)
	# print("labels")
	# print(labels)
	"""计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
	pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
	# print("pred")
	# print(pred)
	rights = pred.eq(labels.data).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
	# print("rights")
	# print(rights)
	return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素



# 处理数据形成训练数据
# 设置句子的最大长度
MAX_LENGTH = 10

#对英文做标准化处理
# pairs = [[chi, normalizeEngString(eng)] for chi, eng in zip(chinese, english)]
pairs = [[normalizeEngString(eng),chi] for chi, eng in zip(chinese, english)]

# 对句子对做过滤，处理掉那些超过MAX_LENGTH长度的句子
input_lang = Lang('Chinese')
output_lang = Lang('English')
pairs = [pair for pair in pairs if filterPair(pair)]

# 建立两个字典（中文的和英文的）
for pair in pairs:
	input_lang.addSentence(pair[0])
	output_lang.addSentence(pair[1])



# 形成训练集，首先，打乱所有句子的顺序
random_idx = np.random.permutation(range(len(pairs)))
pairs = [pairs[i] for i in random_idx]

# 将语言转变为单词的编码构成的序列
pairs = [indexFromPair(pair) for pair in pairs]

# 形成训练集、校验集和测试集
valid_size = len(pairs) // 10
if valid_size > 10000:
	valid_size = 10000
pairs = pairs[ : - valid_size]
valid_pairs = pairs[-valid_size : -valid_size // 2]
test_pairs = pairs[- valid_size // 2 :]

# 利用PyTorch的dataset和dataloader对象，将数据加载到加载器里面，并且自动分批

batch_size = 30 #一撮包含30个数据记录，这个数字越大，系统在训练的时候，每一个周期处理的数据就越多，这样处理越快，但总的数据量会减少



# 形成训练对列表，用于喂给train_dataset
# 每一类只要一个样本，用于临时研究
pairs_X = [pair[0] for pair in pairs][:6000]
pairs_Y = [pair[1] for pair in pairs][:6000]
valid_X = [pair[0] for pair in pairs][:30]
valid_Y = [pair[1] for pair in pairs][:30]
test_X = [pair[0] for pair in pairs][:400]
test_Y = [pair[1] for pair in pairs][:400]


# 形成训练集
train_dataset = DataSet.TensorDataset(torch.LongTensor(pairs_X), torch.LongTensor(pairs_Y))
# 形成数据加载器
train_loader = DataSet.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=8)

# 校验数据
valid_dataset = DataSet.TensorDataset(torch.LongTensor(valid_X), torch.LongTensor(valid_Y))
valid_loader = DataSet.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers=8)

# 测试数据
test_dataset = DataSet.TensorDataset(torch.LongTensor(test_X), torch.LongTensor(test_Y))
test_loader = DataSet.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers=1):
		super(EncoderRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		# 第一层Embeddeing
		self.embedding = nn.Embedding(input_size, hidden_size)
		# 第二层GRU，注意GRU中可以定义很多层，主要靠num_layers控制
		self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True, num_layers = self.n_layers, bidirectional = True)

	def forward(self,input,hidden):
		#前馈过程
		#input尺寸： batch_size, length_seq
		embedded = self.embedding(input)
		#embedded尺寸：batch_size, length_seq, hidden_size
		output = embedded
		output, hidden = self.gru(output, hidden)
		# output尺寸：batch_size, length_seq, hidden_size
		# hidden尺寸：num_layers * directions, batch_size, hidden_size
		return output,hidden

	def initHidden(self, batch_size):
		# 对隐含单元变量全部进行初始化
		#num_layers * num_directions, batch, hidden_size
		result = Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

# 解码器网络
class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, n_layers=1):
		super(DecoderRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		# 嵌入层
		self.embedding = nn.Embedding(output_size, hidden_size)
		# GRU单元
		# 设置batch_first为True的作用就是为了让GRU接受的张量可以和其它单元类似，第一个维度为batch_size
		self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True,num_layers = self.n_layers, bidirectional = True)
		# 最后的全链接层
		self.out = nn.Linear(hidden_size * 2, output_size)
		self.softmax = nn.LogSoftmax()

	def forward(self, input, hidden):
		# input大小：batch_size, length_seq
		output = self.embedding(input)
		# embedded大小：batch_size, length_seq, hidden_size
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		# output大小：batch_size, length_seq, hidden_size * directions
		# hidden大小：n_layers * directions, batch_size, hidden_size
		output = self.softmax(self.out(output[:, -1, :]))
		# output大小：batch_size * output_size
		# 从output中取时间步重新开始

		return output, hidden
	def initHidden(self):
		# 初始化隐含单元的状态，输入变量的尺寸：num_layers * directions, batch_size, hidden_size
		result = Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

# 开始训练过程
# 定义网络结构
hidden_size = 512
max_length = MAX_LENGTH
n_layers = 1

encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers = n_layers)
decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers = n_layers)

if use_cuda:
	# 如果本机有GPU可用，则将模型加载到GPU上
	encoder = encoder.cuda()
	decoder = decoder.cuda()

learning_rate = 0.001
# 为两个网络分别定义优化器
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

# 定义损失函数
criterion = nn.NLLLoss()
teacher_forcing_ratio = 1

plot_losses = []

# 开始200轮的循环
num_epoch = 1
for epoch in range(num_epoch):
	print_loss_total = 0
	# 对训练数据循环
	for data in train_loader:
		print("training")
		input_variable = Variable(data[0]).cuda() if use_cuda else Variable(data[0])
		# print("input size")
		# print(input_variable.data.size())
		# input_variable的大小：batch_size, length_seq
		target_variable = Variable(data[1]).cuda() if use_cuda else Variable(data[1])
		# print("target size")
		# print(target_variable.data.size())
		
		# target_variable的大小：batch_size, length_seq

		# 初始化编码器状态
		encoder_hidden = encoder.initHidden(data[0].size()[0])
		# print("encoder hidden size")
		# print(encoder_hidden.data.size())
		# 清空梯度
		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()

		loss = 0

		# 开始编码器的计算，对时间步的循环由系统自动完成
		encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
		# print("encoder output size")
		# print(encoder_outputs.data.size())
		# encoder_outputs的大小：batch_size, length_seq, hidden_size*direction
		# encoder_hidden的大小：direction*n_layer, batch_size, hidden_size

		# 开始解码器的工作
		# 输入给解码器的第一个字符
		decoder_input = Variable(torch.LongTensor([[SOS_token]] * target_variable.size()[0]))
		# decoder_input大小：batch_size, length_seq
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

		# 让解码器的隐藏层状态等于编码器的隐藏层状态
		decoder_hidden = encoder_hidden
		# decoder_hidden大小：direction*n_layer, batch_size, hidden_size

		# 以teacher_forcing_ratio的比例用target中的翻译结果作为监督信息
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		base = torch.zeros(target_variable.size()[0])
		print("\n")
		if use_teacher_forcing:
			# 教师监督: 将下一个时间步的监督信息输入给解码器
			# 对时间步循环
			for di in range(MAX_LENGTH):
				# 开始一步解码
				decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
				# print("decoder output size")
				# print(decoder_output.data.size())
				# decoder_ouput大小：batch_size, output_size
				# 计算损失函数
				loss += criterion(decoder_output, target_variable[:, di])
				# print("target variable [:,di]")
				# print(target_variable[:, di].data.size())
				# 将训练数据当做下一时间步的输入
				decoder_input = target_variable[:, di].unsqueeze(1)  # Teacher forcing
				# print("decoder input size")
				# print(decoder_input.data.size())
				# decoder_input大小：batch_size, length_seq
		else:
			# 没有教师训练: 使用解码器自己的预测作为下一时间步的输入
			# 开始对时间步进行循环
			for di in range(MAX_LENGTH):
				# 进行一步解码
				decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
				#decoder_ouput大小：batch_size, output_size(vocab_size)

				#从输出结果（概率的对数值）中选择出一个数值最大的单词作为输出放到了topi中
				topv, topi = decoder_output.data.topk(1, dim = 1)
				#topi 尺寸：batch_size, k
				ni = topi[:, 0]

				# 将输出结果ni包裹成Variable作为解码器的输入
				decoder_input = Variable(ni.unsqueeze(1))
				# decoder_input大小：batch_size, length_seq
				decoder_input = decoder_input.cuda() if use_cuda else decoder_input

				#计算损失函数
				loss += criterion(decoder_output, target_variable[:, di])



		# 开始反向传播
		loss.backward()
		loss = loss.cpu() if use_cuda else loss
		# 开始梯度下降
		encoder_optimizer.step()
		decoder_optimizer.step()
		# 累加总误差
		print_loss_total += loss.data.numpy()[0]

	# 计算训练时候的平均误差
	print_loss_avg = print_loss_total / len(train_loader)
	    
	# 开始跑校验数据集
	valid_loss = 0
	rights = []
	# 对校验数据集循环
	for data in valid_loader:
		print("validating")
		input_variable = Variable(data[0]).cuda() if use_cuda else Variable(data[0])
		# input_variable的大小：batch_size, length_seq
		target_variable = Variable(data[1]).cuda() if use_cuda else Variable(data[1])
		# target_variable的大小：batch_size, length_seq

		encoder_hidden = encoder.initHidden(data[0].size()[0])

		loss = 0
		encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
		# encoder_outputs的大小：batch_size, length_seq, hidden_size*direction
		# encoder_hidden的大小：direction*n_layer, batch_size, hidden_size

		decoder_input = Variable(torch.LongTensor([[SOS_token]] * target_variable.size()[0]))
		# decoder_input大小：batch_size, length_seq
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

		decoder_hidden = encoder_hidden
		# decoder_hidden大小：direction*n_layer, batch_size, hidden_size

		# 没有教师监督: 使用解码器自己的预测作为下一时间步解码器的输入
		for di in range(MAX_LENGTH):
			# 一步解码器运算
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			#decoder_ouput大小：batch_size, output_size(vocab_size)

			# 选择输出最大的项作为解码器的预测答案
			topv, topi = decoder_output.data.topk(1, dim = 1)
			#topi 尺寸：batch_size, k
			ni = topi[:, 0]
			decoder_input = Variable(ni.unsqueeze(1))
			# decoder_input大小：batch_size, length_seq
			decoder_input = decoder_input.cuda() if use_cuda else decoder_input

			# 计算预测的准确率，记录在right中，right为一个二元组，分别存储猜对的个数和总数
			right = rightness(decoder_output, target_variable[:, di].unsqueeze(1))
			rights.append(right)

			# 计算损失函数
			loss += criterion(decoder_output, target_variable[:, di])
		loss = loss.cpu() if use_cuda else loss
		# 累加校验时期的损失函数
		valid_loss += loss.data.numpy()[0]
	# 打印每一个Epoch的输出结果
	right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
	print('进程：%d%% 训练损失：%.4f，校验损失：%.4f，词正确率：%.2f%%' % (epoch * 1.0 / num_epoch * 100, print_loss_avg,valid_loss / len(valid_loader),100.0 * right_ratio))
	# 记录基本统计指标
	plot_losses.append([print_loss_avg, valid_loss / len(valid_loader), right_ratio])



# 首先，在测试集中随机选择20个句子作为测试
indices = np.random.choice(range(len(test_X)), 10)

# 对每个句子进行循环
for ind in indices:
	data = [test_X[ind]]
	target = [test_Y[ind]]
	# 把源语言的句子打印出来
	print(SentenceFromList(input_lang, data[0]))
	input_variable = Variable(torch.LongTensor(data)).cuda() if use_cuda else Variable(torch.LongTensor(data))
	# input_variable的大小：batch_size, length_seq
	target_variable = Variable(torch.LongTensor(target)).cuda() if use_cuda else Variable(torch.LongTensor(target))
	# target_variable的大小：batch_size, length_seq

	# 初始化编码器
	encoder_hidden = encoder.initHidden(input_variable.size()[0])

	loss = 0

	# 编码器开始编码，结果存储到了encoder_hidden中
	encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
	# encoder_outputs的大小：batch_size, length_seq, hidden_size*direction
	# encoder_hidden的大小：direction*n_layer, batch_size, hidden_size

	# 将SOS作为解码器的第一个输入
	decoder_input = Variable(torch.LongTensor([[SOS_token]] * target_variable.size()[0]))
	# decoder_input大小：batch_size, length_seq
	decoder_input = decoder_input.cuda() if use_cuda else decoder_input

	# 将编码器的隐含层单元数值拷贝给解码器的隐含层单元
	decoder_hidden = encoder_hidden
	# decoder_hidden大小：direction*n_layer, batch_size, hidden_size

	# 没有教师指导下的预测: 使用解码器自己的预测作为解码器下一时刻的输入
	output_sentence = []
	decoder_attentions = torch.zeros(max_length, max_length)
	rights = []
	# 按照输出字符进行时间步循环
	for di in range(MAX_LENGTH):
		# 解码器一个时间步的计算
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
		#decoder_ouput大小：batch_size, output_size(vocab_size)

		# 解码器的输出
		topv, topi = decoder_output.data.topk(1, dim = 1)
		#topi 尺寸：batch_size, k
		ni = topi[:, 0]
		decoder_input = Variable(ni.unsqueeze(1))
		ni = ni.numpy()[0]
		# print(ni)
		# 将本时间步输出的单词编码加到output_sentence里面
		output_sentence.append(ni)
		# decoder_input大小：batch_size, length_seq
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

		# 计算输出字符的准确度
		right = rightness(decoder_output, target_variable[:, di].unsqueeze(1))
		rights.append(right)
	# 解析出编码器给出的翻译结果
	sentence = SentenceFromList(output_lang, output_sentence)
	# 解析出标准答案
	standard = SentenceFromList(output_lang, target[0])

	# 将句子打印出来
	print('机器翻译：', sentence)
	print('标准翻译：', standard)
	# 输出本句话的准确率
	right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
	print('词准确率：', 100.0 * right_ratio)
	print('\n')
