# 训练一个神经网络，做到自动将输入的英文词翻译为中文
# 核心思路：使用人工翻译文本，生成中英文两份词语编码，针对一个英文单词，选取中文编码中距离最近的词作为翻译结果。
# 效果不太好，可能是因为训练样本只有两本书，比较少。


# # 数值运算和绘图的程序包
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib


# 加载机器学习的软件包
from sklearn.decomposition import PCA

from gensim.models import Word2Vec

#加载‘结巴’中文分词软件包
import jieba

#加载正则表达式处理的包
import re

# 读取中文和英文的书

f = open("gan-zh.txt", 'r')
lines_zh = []
for line in f:
	temp = jieba.lcut(line)
	words = []
	for i in temp:
		#过滤掉所有的标点符号
		i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
		if len(i) > 0:
			words.append(i)
	if len(words) > 0:
		lines_zh.append(words)

f_en = open("gan-en.txt","r")
lines_en = []
for line in f_en:
	temp = line.split()
	words = []
	for i in temp:
		#过滤掉所有的标点符号
		i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
		if len(i) > 0:
			words.append(i)
	if len(words) > 0:
		lines_en.append(words)

model_zh = Word2Vec(lines_zh, size = 100, window = 5 , min_count = 3)
model_en = Word2Vec(lines_en, size = 100, window = 5 , min_count = 3)



# 定义计算cosine相似度的函数
def cos_similarity(vec1, vec2):
	norm1 = np.linalg.norm(vec1)
	norm2 = np.linalg.norm(vec2)
	norm = norm1 * norm2
	dot = np.dot(vec1, vec2)
	result = dot / norm if norm > 0 else 0
	return result

# 在所有的词向量中寻找到与目标词（word）相近的向量，并按相似度进行排列
def find_most_similar(word, model_origin, model_target):
	vector = model_origin[word]
	simi = []
	for new_word in model_target.wv.vocab:
		simi.append((cos_similarity(vector,model_target[new_word]),new_word))
	simi = sorted(simi,key=lambda x:x[0],reverse=True)
	print(simi[0])

# 展示效果
find_most_similar("快乐",model_zh,model_en)
find_most_similar("moment",model_en,model_zh)
find_most_similar("plane",model_en,model_zh)
find_most_similar("go",model_en,model_zh)
find_most_similar("what",model_en,model_zh)
