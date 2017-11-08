# 分别将中文词向量中的一、二、三、⋯⋯和英文词向量中的 one,two,three,⋯⋯画在二维平面上，并比较二者

# 加载必要的程序包
# # 数值运算和绘图的程序包
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib


# 加载机器学习的软件包
from sklearn.decomposition import PCA

# #加载Word2Vec的软件包
from gensim.models.keyedvectors import KeyedVectors



# 中文的字典
word_vectors = KeyedVectors.load_word2vec_format('vectors.bin', binary=True, unicode_errors='ignore')

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

# 中文的一二三四五列表
cn_list = {'一', '二', '三', '四', '五', '六', '七', '八', '九', '零'}
# 阿拉伯数字的12345列表
en_list = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
# 英文数字的列表
en_list = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero'}


# 对应词向量都存入到列表中
cn_vectors = []  #中文的词向量列表
en_vectors = []  #英文的词向量列表
for w in cn_list:
	cn_vectors.append(word_vectors[w])
for w in en_list:
	en_vectors.append(word_vectors_en[w])


# 将这些词向量统一转化为矩阵
cn_vectors = np.array(cn_vectors)
en_vectors = np.array(en_vectors)

# 降维实现可视化
X_reduced = PCA(n_components=2).fit_transform(cn_vectors)
Y_reduced = PCA(n_components = 2).fit_transform(en_vectors)

# 绘制所有单词向量的二维空间投影
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 8))
ax1.plot(X_reduced[:, 0], X_reduced[:, 1], 'o')
ax2.plot(Y_reduced[:, 0], Y_reduced[:, 1], 'o')
zhfont1 = matplotlib.font_manager.FontProperties(fname='/Library/Fonts/华文仿宋.ttf', size=16)
for i, w in enumerate(cn_list):
	ax1.text(X_reduced[i, 0], X_reduced[i, 1], w, fontproperties = zhfont1, alpha = 1)
for i, w in enumerate(en_list):
	ax2.text(Y_reduced[i, 0], Y_reduced[i, 1], w, alpha = 1)

plt.show()