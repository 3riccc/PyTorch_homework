# 运用Google训练好的英文词向量:
# 验证:man-woman=king-queen, man-woman=son-daughter, 等


# 加载必要的程序包
import numpy as np
from numpy import *


# 实现余弦相似度计算
# 传入向量的形式应是1*n numpy array
def euclid_dis(vector1,vector2):
	vector1 = vector1.reshape((1,-1))
	vector2 = vector2.reshape((1,-1))
	fenmu = sqrt((vector1 * vector1).sum()) * sqrt((vector2 * vector2).sum())
	if fenmu == 0:
		return 100000
	else:
		return (np.matmul((vector1-vector2),(vector1-vector2).T).squeeze()) / fenmu


# 实现most similar，找出最相关的词
# positive array lenth = 2
# negative array lenth = 2
def most_similar(word_vector,positive=[],negative=[]):
	word_wanted = ''
	min_len = 1000000
	positive_dis = word_vector[positive[0]] - word_vector[positive[1]]
	start_point = word_vector[negative[0]]
	for i,word in enumerate(word_vector):
		cos_dis = euclid_dis((start_point - word_vector[word]),positive_dis)
		if cos_dis < min_len:
			min_len = cos_dis
			word_wanted = word
	print(positive[0] + " - "+positive[1]+" == "+negative[0] + " - "+word_wanted)
	return word_wanted


# 将英文的词向量都存入如下的字典中
# 从老师的源文件中抄过来
f = open('./glove.6B/glove.6B.50d.txt', 'r')
i = 1
word_vectors_en = {}
with open('./glove.6B/glove.6B.50d.txt') as f:
	for line in f:
		numbers = line.split()
		word = numbers[0]
		vectors = np.array([float(i) for i in numbers[1 : ]])
		word_vectors_en[word] = vectors
		i += 1

most_similar(word_vectors_en,['man','king'],['woman'])
most_similar(word_vectors_en,['man','son'],['woman'])
most_similar(word_vectors_en,['man','actor'],['woman'])

