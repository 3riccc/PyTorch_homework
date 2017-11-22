# 训练一个神经网络，做到自动将输入的英文词翻译为中文
# 核心思路：使用人工翻译文本，生成中英文两份词语编码，针对一个英文单词，选取中文编码中距离最近的词作为翻译结果。
# 效果不太好，可能是因为训练样本只有两本书，比较少。
import os

# # 数值运算和绘图的程序包
import numpy as np
from numpy import *

# 加载Word2Vec
from gensim.models import Word2Vec

#加载‘结巴’中文分词软件包
import jieba

#加载正则表达式处理的包
import re

# 读取中文和英文的书
def getDirFiles(paths,origin_path):
	if os.path.isfile(origin_path):
	    paths.append(origin_path)
	    return paths
	elif os.path.isdir(origin_path):
		dir_list = os.listdir(origin_path)
		for index in dir_list:
			paths = getDirFiles(paths,origin_path+"/"+index)
	return paths

def getLines(book_list,lang):
	lang_list = []
	for book in book_list:
		try:
			f = open(book,'r')
			print("reading the book "+book)
			for line in f:
				if lang == "zh":
					temp = jieba.lcut(line)
				elif lang == "en":
					temp = line.split()
				else:
					print("请输入正确语言，zh/en")
					return
				# 一句话中的单词们
				words = []
				for i in temp:
					for i in temp:
						#过滤掉所有的标点符号
						i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：]+","",i)
						if len(i) > 0:
							words.append(i)
				lang_list.append(words)

		except UnicodeDecodeError:
			print(book+" can not be readed")
			continue
	return lang_list


try:
	model_zh = Word2Vec.load('model_zh')
	model_en = Word2Vec.load('model_en')
	print("model loaded")
except FileNotFoundError:
	book_list_zh = getDirFiles([],"res-zh")
	book_list_en = getDirFiles([],"res-en")
	lines_zh = getLines(book_list_zh,'zh')
	lines_en = getLines(book_list_en,'en')

	print("generatinging vectors(zh)")
	model_zh = Word2Vec(lines_zh, size = 100, window = 5 , min_count = 3)
	model_zh.save('model_zh')
	print("generatinging vectors(en)")
	model_en = Word2Vec(lines_en, size = 100, window = 5 , min_count = 3)
	model_en.save('model_en')



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
	print("\n")
	print(word)
	vector = model_origin[word]
	simi = []
	for new_word in model_target.wv.vocab:
		simi.append((cos_similarity(vector,model_target[new_word]),new_word))
	simi = sorted(simi,key=lambda x:x[0],reverse=True)
	print(simi[:10])

# 展示效果
# find_most_similar("快乐",model_zh,model_en)
# find_most_similar("moment",model_en,model_zh)
# find_most_similar("",model_en,model_zh)
# find_most_similar("life",model_en,model_zh)
find_most_similar("steam",model_en,model_zh)
find_most_similar("eat",model_en,model_zh)
